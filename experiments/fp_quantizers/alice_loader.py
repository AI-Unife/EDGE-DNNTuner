"""
Loader for the pre-trained CIFAR-10 models from Alice (.keras, Keras 2.15).

These files were saved as a Sequential nested in a Sequential and `tf.keras.models.load_model` fails
on them with "Layer 'conv2d' expected 2 variables, but received 0 variables". The archive itself is
fine: config.json holds a flat list of the layers and model.weights.h5 holds the weights under
layers/<layer name>/vars/<i>. So we rebuild the model layer by layer and copy the weights by name.

    from alice_loader import load_alice_model
    model = load_alice_model(path)          # compiled Keras model (categorical_crossentropy, accuracy)
"""
import io
import json
import zipfile

import h5py
import numpy as np
import tensorflow as tf


def _flat_layers(config):
    """The layers of the network in order, skipping the wrapper Sequential and input layers."""
    out = []
    for l in config["config"]["layers"]:
        if l["class_name"] in ("Sequential", "InputLayer"):
            continue
        out.append(l)
    return out


def load_alice_model(path, verbose=False):
    """Standard Keras loading first (works for Functional models, e.g. the HW3 ones with skip
    connections); the manual rebuild below only for the nested-Sequential files that fail."""
    try:
        model = tf.keras.models.load_model(path, compile=False)
        if verbose:
            print("loaded with tf.keras.models.load_model")
    except Exception as e:   # noqa: BLE001 - any loading failure -> manual rebuild
        print(f"  [alice_loader] load_model failed ({type(e).__name__}: {str(e)[:100]}...) -> rebuilding from config + weights")
        with zipfile.ZipFile(path) as z:
            class_name = json.loads(z.read("config.json"))["class_name"]
        if class_name == "Functional":
            model = _rebuild_functional(path, verbose)
        else:
            model = _rebuild_sequential(path, verbose)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def _rebuild_sequential(path, verbose=False):
    with zipfile.ZipFile(path) as z:
        config = json.loads(z.read("config.json"))
        h5_bytes = z.read("model.weights.h5")

    layers_cfg = _flat_layers(config)
    # input shape: from the first InputLayer found in the config
    input_shape = None
    for l in config["config"]["layers"]:
        if l["class_name"] == "InputLayer":
            shape = l["config"].get("batch_input_shape") or l["config"].get("batch_shape")
            if shape:
                input_shape = tuple(shape[1:])
                break
    if input_shape is None:
        input_shape = (32, 32, 3)

    model = tf.keras.Sequential(name="alice_model")
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    for l in layers_cfg:
        layer = tf.keras.layers.deserialize(l)   # full dict (module, class_name, config)
        model.add(layer)

    n_set = _load_weights_by_name(model, h5_bytes, verbose)
    if verbose:
        print(f"loaded {n_set} weight tensors into {len(model.layers)} layers")
    return model


def _load_weights_by_name(model, h5_bytes, verbose=False):
    """Copy the weights of every layer from model.weights.h5 (layers/<name>/vars/<i>).
    Keras 2.15 sometimes stores the weights of a layer under the name of ANOTHER layer of the same
    type (e.g. conv2d_1 <-> conv2d_2 in the HW3 models). So: match by name when the shapes agree,
    otherwise by the (unique) list of weight shapes among the entries not used yet."""
    n_set = 0
    with h5py.File(io.BytesIO(h5_bytes), "r") as h:
        entries = {}   # h5 layer name -> list of arrays
        for name in h["layers"]:
            g = h["layers"][name]
            if "vars" in g and len(g["vars"]):
                entries[name] = [g["vars"][k][()] for k in sorted(g["vars"].keys(), key=int)]
    used = set()
    for layer in model.layers:
        if not layer.weights:
            continue
        shapes_model = [tuple(w.shape) for w in layer.weights]
        chosen = None
        if layer.name in entries and layer.name not in used and [v.shape for v in entries[layer.name]] == shapes_model:
            chosen = layer.name
        else:
            cands = [n for n, vals in entries.items() if n not in used and [v.shape for v in vals] == shapes_model]
            if len(cands) == 1:
                chosen = cands[0]
            elif not cands:
                raise ValueError(f"layer '{layer.name}': no weights with shapes {shapes_model} in model.weights.h5")
            else:
                raise ValueError(f"layer '{layer.name}': ambiguous weights, candidates {cands} (shapes {shapes_model})")
        layer.set_weights(entries[chosen])
        used.add(chosen)
        n_set += len(entries[chosen])
        if verbose:
            tag = "" if chosen == layer.name else f"   (stored under '{chosen}')"
            print(f"  {layer.name:24s} <- {shapes_model}{tag}")
    return n_set


def _rebuild_functional(path, verbose=False):
    """Rebuild a Functional model (skip connections, Add, BatchNorm...) from config.json by following
    the inbound_nodes of every layer, then copy the weights by layer name."""
    with zipfile.ZipFile(path) as z:
        config = json.loads(z.read("config.json"))
        h5_bytes = z.read("model.weights.h5")
    cfg = config["config"]
    tensors = {}
    for l in cfg["layers"]:
        name = l["config"]["name"]
        if l["class_name"] == "InputLayer":
            shape = l["config"].get("batch_input_shape") or l["config"].get("batch_shape")
            tensors[name] = tf.keras.Input(shape=tuple(shape[1:]), name=name)
            continue
        if l["class_name"] == "BatchNormalization" and isinstance(l["config"].get("axis"), list) and len(l["config"]["axis"]) == 1:
            l["config"]["axis"] = l["config"]["axis"][0]   # Keras 2 saves [3], Keras 3 wants 3
        layer = tf.keras.layers.deserialize(l)
        # inbound_nodes: [[[src_name, node_idx, tensor_idx, kwargs], ...]]  (Keras 2 format)
        node = l["inbound_nodes"][0]
        if isinstance(node, dict):                      # Keras 3 format: {"args": [...], ...}
            raise ValueError("Keras 3 inbound_nodes format not supported by this loader")
        srcs = [n[0] for n in node]
        inputs = [tensors[s_] for s_ in srcs]
        tensors[name] = layer(inputs if len(inputs) > 1 else inputs[0])
    ins = [tensors[i[0]] for i in cfg["input_layers"]]
    outs = [tensors[o[0]] for o in cfg["output_layers"]]
    model = tf.keras.Model(ins if len(ins) > 1 else ins[0], outs if len(outs) > 1 else outs[0], name="alice_model")
    n_set = _load_weights_by_name(model, h5_bytes, verbose)
    if verbose:
        print(f"loaded {n_set} weight tensors into {len(model.layers)} layers (functional)")
    return model


if __name__ == "__main__":
    import sys
    m = load_alice_model(sys.argv[1], verbose=True)
    m.summary()
