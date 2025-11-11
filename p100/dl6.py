# Minimal MobileNetV2 transfer learning (local dataset) - a -> e
# Clean and simple (no config block, no visualization)

import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------- change only these paths ----------
TRAIN_DIR = "dataset/train"
VAL_DIR   = "dataset/val"
# ---------------------------------------------

# ---------- a) Load local data (simple rescaling only) ----------
train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    TRAIN_DIR,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    VAL_DIR,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_gen.class_indices)
print("Detected classes:", train_gen.class_indices)

# ---------- b) Load pre-trained MobileNetV2 and freeze base ----------
base_model = MobileNetV2(weights='imagenet', include_top=False,
                         input_shape=(64, 64, 3), pooling='avg')
base_model.trainable = False  # freeze the base model

# ---------- c) Add a custom classifier ----------
inputs = layers.Input(shape=(64, 64, 3))
x = base_model(inputs, training=False)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs, name="mobilenetv2_transfer_simple")
model.summary()

# ---------- d) Train the new classifier head ----------
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\nTraining new classifier head...")
model.fit(train_gen, validation_data=val_gen, epochs=6)

# ---------- e) Fine-tune (unfreeze last 20 layers) ----------
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\nFine-tuning last 20 layers...")
model.fit(train_gen, validation_data=val_gen, epochs=3)

# ---------- Save and evaluate ----------
model.save("mobilenetv2_final_simple.h5")
print("Saved model as 'mobilenetv2_final_simple.h5'")

loss, acc = model.evaluate(val_gen, verbose=1)
print(f"Validation loss: {loss:.4f}  -  Validation accuracy: {acc:.4f}")