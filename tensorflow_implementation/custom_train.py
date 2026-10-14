from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train_model(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    train_data,  # Can be array or [data, pos] list
    train_labels: tf.Tensor,
    test_data,   # Can be array or [data, pos] list
    test_labels: tf.Tensor,
    epochs: int,
    params: Dict,
    callbacks: List[tf.keras.callbacks.Callback],
) -> Dict[str, List[float]]:
    """
    Custom training loop that mimics `model.fit` structure:
      - supports Keras callbacks (EarlyStopping, ReduceLROnPlateau, TensorBoard, ...)
      - returns a `history` dict like `model.fit(...).history`
      - handles both regular and ROI (dual-input) datasets

    Expected shapes:
      - train_data: [B, T, ...] or [data_array, pos_array]
      - train_labels: [B, T, C] (one-hot)
      - test_data:  [B_val, T, ...] or [data_array, pos_array]
      - test_labels:[B_val, T, C] (one-hot)

    Notes:
      - The model outputs softmax probabilities: CategoricalCrossentropy(from_logits=False).
      - Sequence accuracy is a soft majority vote (per-frame probabilities averaged over time).
      - Per-epoch validation runs eval_model on the full validation set.
    """
    # -------------------------------------------------------------------------
    # Data pipeline (simple & deterministic; enable shuffle if needed)
    # -------------------------------------------------------------------------
    batch_size = int(params["batch_size"])

    # Check if this is a ROI dataset (list with two arrays)
    is_roi = isinstance(train_data, list) and len(train_data) == 2

    if is_roi:
        # ROI case: train_data = [data, pos]
        data_array, pos_array = train_data
        print(f"Creating dataset with ROI inputs: data shape={data_array.shape}, pos shape={pos_array.shape}")
        # Create dataset where each sample is ((data_frame, pos_frame), label)
        # So when iterating, we get batch_input = (batch_data, batch_pos) and batch_y
        ds = tf.data.Dataset.from_tensor_slices(((data_array, pos_array), train_labels))
    else:
        # Regular case: each sample is (data_frame, label)
        ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    
    # Uncomment to enable shuffling & prefetch for performance:
    # ds = ds.shuffle(buffer_size=min(4 * batch_size, 1000), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    # ds = ds.prefetch(tf.data.AUTOTUNE)

    # -------------------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------------------
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    @tf.function(reduce_retracing=True)
    def train_step(batch_input, batch_y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        One training step:
          - unroll model over time dimension T (frame-wise forward)
          - stack outputs to [B, T, C]
          - compute loss & gradients, apply optimizer
          - compute batch sequence accuracy (soft majority vote)
          
        Args:
            batch_input: For ROI, a tuple (batch_x, batch_pos); otherwise just batch_x
            batch_y: labels [B, T, C]
        """
        # Handle ROI vs regular input
        if is_roi:
            batch_x, batch_pos = batch_input
        else:
            batch_x = batch_input
        
        time_steps = tf.shape(batch_x)[1]

        out_dtype = tf.as_dtype(getattr(model, "compute_dtype", tf.float32))

        with tf.GradientTape() as tape:
            ta = tf.TensorArray(dtype=out_dtype, size=time_steps)

            # Temporal unroll; using tf.range inside @tf.function creates a tf.while_loop but with TensorArray it's ok
            for t in tf.range(time_steps):
                # batch_x[:, t] : [B, ...]
                if is_roi:
                    # For ROI: pass both data and pos
                    out_t = model([batch_x[:, t], batch_pos[:, t]], training=True)  # [B, C]
                else:
                    out_t = model(batch_x[:, t], training=True)  # [B, C]
                ta = ta.write(t, out_t)

            # ta.stack(): [T, B, C] -> permuta a [B, T, C]
            outputs = tf.transpose(ta.stack(), perm=[1, 0, 2])

            loss = loss_fn(batch_y, outputs)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Soft majority vote: average the per-frame probabilities over time.
        # The label is constant over time, so frame 0 carries it.
        seq_pred = tf.argmax(tf.reduce_mean(outputs, axis=1), axis=1, output_type=tf.int32)
        seq_true = tf.argmax(batch_y[:, 0, :], axis=1, output_type=tf.int32)
        batch_acc = tf.reduce_mean(tf.cast(tf.equal(seq_pred, seq_true), tf.float32))

        return loss, batch_acc

    # -------------------------------------------------------------------------
    # History & callbacks
    # -------------------------------------------------------------------------
    history = {key: [] for key in ["loss", "accuracy", "val_loss", "val_accuracy"]}

    for cb in callbacks:
        cb.set_model(model)
        cb.on_train_begin()

    # -------------------------------------------------------------------------
    # Epoch loop
    # -------------------------------------------------------------------------
    for epoch in range(epochs):
        epoch_loss_sum = 0.0
        epoch_acc_sum = 0.0
        n_batches = 0

        for cb in callbacks:
            cb.on_epoch_begin(epoch)

        print(f"Epoch {epoch + 1}/{epochs}")

        # Training
        for batch_x, batch_y in tqdm(ds, leave=False):
            loss_t, acc_t = train_step(batch_x, batch_y)
            epoch_loss_sum += float(loss_t.numpy())
            epoch_acc_sum += float(acc_t.numpy())
            n_batches += 1

        # Aggregate epoch stats
        avg_loss = epoch_loss_sum / max(n_batches, 1)
        avg_acc = epoch_acc_sum / max(n_batches, 1)
        history["loss"].append(avg_loss)
        history["accuracy"].append(avg_acc)

        # Validation
        val_loss, val_acc = eval_model(model, test_data, test_labels)

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}"
        )

        # Feed logs to callbacks (EarlyStopping/ReduceLROnPlateau/etc.)
        logs = {"loss": avg_loss, "accuracy": avg_acc, "val_loss": val_loss, "val_accuracy": val_acc}
        for cb in callbacks:
            cb.on_epoch_end(epoch, logs)
        
        # Check if any callback (e.g., EarlyStopping) requested to stop training
        if model.stop_training:
            print(f"Stopping training at epoch {epoch + 1}/{epochs} (EarlyStopping triggered)")
            break

    for cb in callbacks:
        cb.on_train_end()

    return history  # same shape/keys as Keras History.history


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def eval_model(model: tf.keras.Model, X, y, batch_frames: int = 1024) -> Tuple[float, float]:
    """
    Sequence-level evaluation of a per-frame classifier.

      - every frame of every recording goes through the model (in chunks of
        `batch_frames` frames) -> per-frame probabilities [N, T, C]
      - sequence prediction = soft majority vote: argmax of the per-frame
        probabilities averaged over time
      - loss = CategoricalCrossentropy of each frame vs the recording label,
        averaged over all frames

    Args:
        model: per-frame Keras classifier (softmax output).
        X: [N, T, H, W, C] frames, or [data, pos] for ROI (pos is [N, T, ...]).
        y: recording labels as [N], [N, C] one-hot, or [N, T, C] time-repeated one-hot.

    Returns:
        (mean_loss, sequence_accuracy)
    """
    if isinstance(X, list) and len(X) == 2:
        X_data, X_pos = X
    else:
        X_data, X_pos = X, None

    y = np.asarray(y)
    if y.ndim == 3:
        y_true = y[:, 0, :].argmax(axis=1)
    elif y.ndim == 2:
        y_true = y.argmax(axis=1)
    else:
        y_true = y.astype(int)
    n_classes = int(model.output_shape[-1])

    # Flatten recordings x time into one frame axis: [N*T, ...]
    data = np.stack([np.asarray(x, dtype=np.float32) for x in X_data])
    n_rec, n_t = data.shape[:2]
    frames = data.reshape(n_rec * n_t, *data.shape[2:])
    pos_frames = None
    if X_pos is not None:
        pos = np.stack([np.asarray(p, dtype=np.float32) for p in X_pos])
        pos_frames = pos.reshape(n_rec * n_t, *pos.shape[2:])

    chunks = []
    for start in range(0, len(frames), batch_frames):
        stop = start + batch_frames
        inputs = [frames[start:stop], pos_frames[start:stop]] if pos_frames is not None else frames[start:stop]
        chunks.append(np.asarray(model(inputs, training=False)))
    probs = np.concatenate(chunks).reshape(n_rec, n_t, n_classes)        # [N, T, C]

    preds = probs.mean(axis=1).argmax(axis=1)
    y_frames = np.repeat(np.eye(n_classes, dtype=np.float32)[y_true][:, None, :], n_t, axis=1)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(y_frames, probs)

    return float(loss.numpy()), float((preds == y_true).mean())
