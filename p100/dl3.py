# Simple Image Classification using CNN (Local Dataset)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ------------------------
# 1. Load and preprocess data
# ------------------------
train_dir = 'dataset/train'   # path to training data
test_dir = 'dataset/test'     # path to testing data

train_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='sparse'
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='sparse'
)

# ------------------------
# 2. Define the model
# ------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ------------------------
# 3. Train the model
# ------------------------
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=5
)

# ------------------------
# 4. Evaluate the model
# ------------------------
loss, acc = model.evaluate(test_data)
print("Test Accuracy:", acc)

# ------------------------
# 5. Plot loss and accuracy
# ------------------------
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Curve')
plt.show()