import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# ==========================
# CONFIG
# ==========================

# Path to your saved model
MODEL_PATH = "leaf_mobilenetv2_best.h5"

# change if your file has another name

# Path to the image you want to test
# You can change this to "b.jpeg", "t2.jpg", etc.
IMAGE_PATH = "grey.jpeg"


# Class names MUST match the order of class_indices from training.
# Example: if train_generator.class_indices was:
# {'Blight': 0, 'Common_Rust': 1, 'Gray_Leaf_Spot': 2, 'Healthy': 3}
# then:
class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# This will be filled automatically from model.input_shape
IMG_SIZE = None  # (height, width)


# ==========================
# FUNCTIONS
# ==========================

def load_model():
    """
    Load the trained Keras model and set the global IMG_SIZE
    based on the model's expected input shape.
    """
    global IMG_SIZE

    print("Loading model from:", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)

    # Get model input shape: (None, height, width, channels)
    input_shape = model.input_shape
    print("Model expects input shape:", input_shape)

    if len(input_shape) != 4:
        raise ValueError(f"Unexpected model.input_shape: {input_shape}")

    height = input_shape[1]
    width = input_shape[2]
    IMG_SIZE = (height, width)

    print(f"Using IMG_SIZE for prediction: {IMG_SIZE}")
    return model


def prepare_image(img_path):
    """
    Load and preprocess image to match the training preprocessing:
    - resize to IMG_SIZE
    - convert to array
    - scale pixel values to [0, 1] (same as rescale=1./255)
    """
    if IMG_SIZE is None:
        raise ValueError("IMG_SIZE is not set. Load the model first.")

    print("Preparing image:", img_path)
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)       # (h, w, 3)
    img_array = img_array / 255.0       # IMPORTANT: must match training
    img_array = np.expand_dims(img_array, axis=0)  # (1, h, w, 3)
    return img_array


def predict_image(model, img_path):
    """
    Run prediction on a single image and return:
    - predicted class name
    - confidence
    """
    img = prepare_image(img_path)
    preds = model.predict(img)

    # Show raw probabilities for debugging
    print("Raw prediction vector:", preds[0])

    predicted_idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))
    disease = class_names[predicted_idx]
    return disease, confidence


# ==========================
# MAIN
# ==========================

if __name__ == "__main__":
    print(">>> Running predict.py (CNN version with auto IMG_SIZE)")
    model = load_model()
    disease, confidence = predict_image(model, IMAGE_PATH)
    print(f"Predicted Disease: {disease}")
    print(f"Confidence: {confidence:.2f}")
