# train_finetune.py
import os
import math
import json
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------
# Config
# ---------------------------
DATA_DIR = "dataset/train"   # should contain class subfolders
VAL_DIR = "dataset/val"      # optionally separate val
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
NUM_CLASSES = 4              # adjust if different
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 20
UNFREEZE_AT = 100           # layer index to unfreeze (tune)
MODEL_NAME = "leaf_mobilenetv2_best"
SAVE_DIR = "./models"

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------
# Create datasets
# ---------------------------
train_ds = image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=42
)

val_ds = image_dataset_from_directory(
    VAL_DIR,
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
print("Classes:", class_names)

# Prefetch + autotune
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ---------------------------
# Data augmentation (on GPU/CPU)
# ---------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.12),
    layers.RandomZoom(0.12),
    layers.RandomContrast(0.12),
    layers.RandomTranslation(0.05, 0.05),
])

# ---------------------------
# Build model
# ---------------------------
base_model = MobileNetV2(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False  # stage 1: freeze

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs, outputs)
model.summary()

# ---------------------------
# Class weights (if imbalance)
# ---------------------------
# compute class weights from training directory labels
y = []
for batch_labels in train_ds.unbatch().map(lambda x,y: y):
    # this is a streaming mapping — easier to compute using filenames approach,
    # so we compute class weights from directory counts:
    pass

# Alternative: compute class counts from folders
counts = []
for cls in class_names:
    folder = os.path.join(DATA_DIR, cls)
    counts.append(len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]))
print("Class counts:", dict(zip(class_names, counts)))
labels = []
for i, c in enumerate(class_names):
    labels += [i] * counts[i]
class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# ---------------------------
# Compile + callbacks
# ---------------------------
initial_lr = 1e-3
opt = optimizers.Adam(learning_rate=initial_lr)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

ckpt = callbacks.ModelCheckpoint(
    filepath=os.path.join(SAVE_DIR, MODEL_NAME + ".h5"),
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
early = callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
tb = callbacks.TensorBoard(log_dir="./logs")

# ---------------------------
# Stage 1: train head (frozen base)
# ---------------------------
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    class_weight=class_weights,
    callbacks=[ckpt, reduce_lr, early, tb]
)

# ---------------------------
# Stage 2: fine-tune last layers
# ---------------------------
base_model.trainable = True

# Freeze all layers before UNFREEZE_AT
for layer in base_model.layers[:UNFREEZE_AT]:
    layer.trainable = False
for layer in base_model.layers[UNFREEZE_AT:]:
    layer.trainable = True

# recompile with lower lr
model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    class_weight=class_weights,
    callbacks=[ckpt, reduce_lr, early, tb]
)

# ---------------------------
# Save best model (SavedModel format)
# ---------------------------
best_h5 = os.path.join(SAVE_DIR, MODEL_NAME + ".h5")
print("Best H5 saved at:", best_h5)

saved_dir = os.path.join(SAVE_DIR, MODEL_NAME + "_saved")
print("Saving SavedModel to:", saved_dir)
model.save(saved_dir, save_format="tf")
print("SavedModel saved.")
# Optional: also save labels
with open(os.path.join(SAVE_DIR, "classes.json"), "w") as f:
    json.dump(class_names, f)
