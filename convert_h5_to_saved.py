"""Robust HDF5 (.h5) -> SavedModel converter.

Strategy:
 - Try `tf.keras.models.load_model(h5, compile=False)` first (works if file stores full model).
 - If that fails, try to reconstruct MobileNetV2 head and load weights by name (fallback in `load_weights_from_h5.py`).
 - Save final SavedModel into `models/leaf_mobilenetv2_best_saved` (app.py expects models/leaf_mobilenetv2_best_saved).
"""
import os
import traceback

DEFAULT_H5 = "leaf_mobilenetv2_best.h5"
DEFAULT_OUT = os.path.join("models", "leaf_mobilenetv2_best_saved")


def find_h5():
    # prefer models/ location
    candidates = [os.path.join("models", DEFAULT_H5), DEFAULT_H5]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def save_model_tf(model, out_dir=DEFAULT_OUT):
    print("Saving SavedModel to:", out_dir)
    os.makedirs(out_dir, exist_ok=True)
    model.save(out_dir, save_format="tf")
    print("Saved SavedModel ->", out_dir)
    return True


def main():
    h5 = find_h5()
    if not h5:
        print("Could not find HDF5 file. Looked for:", DEFAULT_H5, "and models/" + DEFAULT_H5)
        return False

    print("Using H5 file:", h5)

    # Attempt 1: direct tf.keras load
    try:
        import tensorflow as tf
        print("Trying tf.keras.models.load_model(...) (compile=False)")
        model = tf.keras.models.load_model(h5, compile=False)
        print("Loaded model via tf.keras.load_model. Saving SavedModel.")
        if save_model_tf(model):
            return True
        return False
    except Exception:
        print("tf.keras.models.load_model failed, traceback:")
        traceback.print_exc()

    # Attempt 2: fallback to weight-based loading using helper script
    try:
        print("Attempting fallback: reconstruct architecture and load weights by name.")
        import load_weights_from_h5 as lw

        lw.inspect_h5(h5)

        m = lw.try_load_weights_by_name(h5, lw.NUM_CLASSES, lw.IMG_SIZE)
        if m is not None:
            if lw.save_if_ok(m, DEFAULT_OUT):
                return True

        m = lw.try_load_weights_full(h5, lw.NUM_CLASSES, lw.IMG_SIZE)
        if m is not None:
            if lw.save_if_ok(m, DEFAULT_OUT):
                return True

        print("All automated attempts failed. See output above for HDF5 contents and errors.")
        return False
    except Exception:
        print("Fallback conversion failed; traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
