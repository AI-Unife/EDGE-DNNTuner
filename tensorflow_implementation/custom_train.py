from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def accuracy(outputs: tf.Tensor, targets: tf.Tensor) -> float:
    """
    Sequence-level accuracy by majority vote (mode) over time.

    Given logits/probabilities over classes for each time-step, this computes:
      - per-frame argmax (predicted class per time-step)
      - per-sequence MODE over time-steps
      - compare the per-sequence predicted mode vs target mode
      - return the average accuracy over the batch (Python float)

    Args:
        outputs: Tensor of shape [B, T, C] (logits or probs are fine for argmax).
        targets: One-hot labels of shape [B, T, C].

    Returns:
        float: mean accuracy across the batch.
    """
    # [B, T]
    pred_frames = tf.argmax(outputs, axis=2, output_type=tf.int32)
    targ_frames = tf.argmax(targets, axis=2, output_type=tf.int32)

    num_classes = tf.shape(outputs)[-1]

    def row_mode(row: tf.Tensor) -> tf.Tensor:
        """Mode (most frequent value) for 1-D int tensor."""
        counts = tf.math.bincount(row, minlength=num_classes, maxlength=num_classes)
        return tf.argmax(counts, axis=0, output_type=tf.int32)

    # [B]
    pred_mode = tf.map_fn(row_mode, pred_frames, fn_output_signature=tf.int32)
    targ_mode = tf.map_fn(row_mode, targ_frames, fn_output_signature=tf.int32)

    acc = tf.reduce_mean(tf.cast(tf.equal(pred_mode, targ_mode), tf.float32))
    return float(acc.numpy())


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
      - Uses CategoricalCrossentropy(from_logits=True).
      - Per-epoch validation runs on the full validation tensor (no dataset).
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
    # Loss & (optional) grad clipping
    # -------------------------------------------------------------------------
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    clipnorm: Optional[float] = params.get("clipnorm")
    clipvalue: Optional[float] = params.get("clipvalue")

    @tf.function(reduce_retracing=True)
    def train_step(batch_input, batch_y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        One training step:
          - unroll model over time dimension T (frame-wise forward)
          - stack outputs to [B, T, C]
          - compute loss & gradients, apply optimizer (with optional clipping)
          - compute batch accuracy (Python conversion is avoided inside tf.function)
          
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

        if clipnorm is not None:
            grads = [tf.clip_by_norm(g, clipnorm) if g is not None else None for g in grads]
        if clipvalue is not None:
            grads = [tf.clip_by_value(g, -clipvalue, clipvalue) if g is not None else None for g in grads]

        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # accuracy() returns a Python float; compute a TF tensor here to avoid pyfunc inside tf.function
        # (We replicate the logic in TF to keep this callable as a graph function.)
        pred_frames = tf.argmax(outputs, axis=2, output_type=tf.int32)
        targ_frames = tf.argmax(batch_y, axis=2, output_type=tf.int32)
        num_classes_local = tf.shape(outputs)[-1]

        def row_mode_tf(row: tf.Tensor) -> tf.Tensor:
            counts = tf.math.bincount(row, minlength=num_classes_local, maxlength=num_classes_local)
            return tf.argmax(counts, axis=0, output_type=tf.int32)

        pred_mode = tf.map_fn(row_mode_tf, pred_frames, fn_output_signature=tf.int32)
        targ_mode = tf.map_fn(row_mode_tf, targ_frames, fn_output_signature=tf.int32)
        batch_acc = tf.reduce_mean(tf.cast(tf.equal(pred_mode, targ_mode), tf.float32))

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

        # Validation — per recording (handles a fixed-Δt / variable-length test set)
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

def eval_model(model: tf.keras.Model, X, y) -> Tuple[float, float]:
    """
    Frame-by-frame evaluation, one recording at a time (handles a per-sample
    number of frames T_i). Dense [B, T, ...] input also works — each row is
    just iterated as a [T, ...] stack.

      - run the model on the [T_i, H, W, C] stack -> [T_i, C] logits
      - sequence prediction = majority vote over per-frame argmax
      - sequence loss = CategoricalCrossentropy with the clip label over T_i frames

    Args:
        model: per-frame Keras classifier (logits output).
        X: object/dense array of [T_i, H, W, C] frames, or [data, pos] for ROI.
        y: clip labels as [B], [B, C] one-hot, or [B, T, C] time-repeated one-hot.

    Returns:
        (mean_loss, sequence_accuracy)
    """
    if isinstance(X, list) and len(X) == 2:
        X_data, X_pos = X
        is_roi = True
    else:
        X_data, X_pos, is_roi = X, None, False

    y = np.asarray(y)
    if y.ndim == 3:
        y_true = y[:, 0, :].argmax(axis=1)
        n_classes = int(y.shape[2])
    elif y.ndim == 2:
        y_true = y.argmax(axis=1)
        n_classes = int(y.shape[1])
    else:
        y_true = y.astype(int)
        n_classes = int(model.output_shape[-1])

    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    preds = np.empty(len(X_data), dtype=int)
    losses = np.empty(len(X_data), dtype=float)

    for i in range(len(X_data)):
        xi = tf.convert_to_tensor(np.asarray(X_data[i], dtype=np.float32))   # [T_i, H, W, C]
        if is_roi:
            pi = tf.convert_to_tensor(np.asarray(X_pos[i], dtype=np.float32))
            out_i = model([xi, pi], training=False)
        else:
            out_i = model(xi, training=False)
        out_i = tf.convert_to_tensor(out_i)                                 # [T_i, C]

        frame_pred = tf.argmax(out_i, axis=-1, output_type=tf.int32).numpy()
        preds[i] = np.bincount(frame_pred, minlength=n_classes).argmax()

        yi = tf.one_hot(np.full(int(out_i.shape[0]), y_true[i], dtype=np.int32), n_classes)
        losses[i] = float(loss_fn(yi, out_i).numpy())

    return float(losses.mean()), float((preds == y_true).mean())