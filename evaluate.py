# evaluate.py
import os, json
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

MODEL_DIR = "models/leaf_mobilenetv2_best_saved"  # saved model dir
VAL_DIR = "dataset/val"
IMG_SIZE = (224,224)
BATCH_SIZE = 32

model = tf.keras.models.load_model(MODEL_DIR)
val_ds = image_dataset_from_directory(VAL_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False, label_mode='int')
class_names = val_ds.class_names

y_true = []
y_pred = []
for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy().tolist())
    y_pred.extend(np.argmax(preds, axis=1).tolist())

print(classification_report(y_true, y_pred, target_names=class_names))
cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:\n", cm)

# simple plot
import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
plt.imshow(cm, cmap='Blues')
plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.yticks(range(len(class_names)), class_names)
plt.colorbar()
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
print("Saved models/confusion_matrix.png")
