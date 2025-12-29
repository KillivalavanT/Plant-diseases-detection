# load_weights_from_h5.py
import os, traceback
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

H5_FILE = "leaf_mobilenetv2_best.h5"   # change if different
# prefer the copy stored under models/ when present
if os.path.exists(os.path.join("models", H5_FILE)):
    H5_FILE = os.path.join("models", H5_FILE)
# default output location used by app.py
OUT_DIR = os.path.join("models", "leaf_mobilenetv2_best_saved")
IMG_SIZE = (224, 224)   # change if you trained with a different size
NUM_CLASSES = 4         # change if your model had different number of classes

def inspect_h5(h5path):
    print("\n--- Inspecting HDF5 file:", h5path)
    try:
        with h5py.File(h5path, 'r') as f:
            print("Top-level keys:", list(f.keys()))
            if 'model_weights' in f:
                print("'model_weights' group keys (first 40):", list(f['model_weights'].keys())[:40])
            if 'layer_names' in f.attrs:
                print("HDF5 file has attribute 'layer_names' (likely contains weights).")
            if 'model_config' in f.attrs or 'model_config' in f:
                print("HDF5 contains model_config (maybe full model).")
    except Exception as e:
        print("Error inspecting HDF5:", e)


def build_model(num_classes, img_size):
    base = MobileNetV2(input_shape=(img_size[0], img_size[1], 3), include_top=False, weights='imagenet')
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    model = models.Model(inputs=base.input, outputs=outputs)
    return model


def try_load_weights_by_name(h5path, num_classes, img_size):
    print("\n--- Trying to load weights by name into MobileNetV2-based model")
    model = build_model(num_classes, img_size)
    try:
        model.load_weights(h5path, by_name=True)
        print("✅ load_weights(by_name=True) SUCCEEDED.")
        return model
    except Exception as e:
        print("❌ load_weights(by_name=True) failed:", e)
        return None


def try_load_weights_full(h5path, num_classes, img_size):
    print("\n--- Trying full model.load_weights(h5) into new model (less likely)")
    model = build_model(num_classes, img_size)
    try:
        model.load_weights(h5path)
        print("✅ model.load_weights(h5) SUCCEEDED.")
        return model
    except Exception as e:
        print("❌ model.load_weights(h5) failed:", e)
        return None


def save_if_ok(model, outdir):
    print("Saving SavedModel to:", outdir)
    try:
        model.save(outdir, save_format='tf')
        print("Saved SavedModel ->", outdir)
        return True
    except Exception as e:
        print("Saving failed:", e)
        traceback.print_exc()
        return False


def main():
    if not os.path.exists(H5_FILE):
        print("HDF5 file not found:", H5_FILE)
        return

    inspect_h5(H5_FILE)

    # Attempt 1: load weights by name
    m = try_load_weights_by_name(H5_FILE, NUM_CLASSES, IMG_SIZE)
    if m is not None:
        if save_if_ok(m, OUT_DIR):
            return

    # Attempt 2: full load_weights
    m = try_load_weights_full(H5_FILE, NUM_CLASSES, IMG_SIZE)
    if m is not None:
        if save_if_ok(m, OUT_DIR):
            return

    # Attempt 3: See if file contains 'model_weights' group and extract layer names
    print("\n--- Final attempt: show more HDF5 structure for manual debugging")
    try:
        with h5py.File(H5_FILE, 'r') as f:
            def print_group(g, indent=0):
                for k in g:
                    print("  " * indent + str(k))
                    try:
                        if isinstance(g[k], h5py.Group):
                            print_group(g[k], indent+1)
                    except Exception:
                        pass
            print_group(f)
    except Exception as e:
        print("Could not open HDF5 for deep listing:", e)
        traceback.print_exc()

    print("\nAll automated attempts failed. Next options:\n"
          " - If this script found 'model_weights' keys, share that output here so I can map layers.\n"
          " - If you still have the training environment, run a short script to load the model and re-save as SavedModel.\n"
          " - I can provide a custom loader depending on what the HDF5 contains.")

if __name__ == "__main__":
    main()
