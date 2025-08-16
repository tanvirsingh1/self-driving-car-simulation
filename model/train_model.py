import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam

warnings.filterwarnings('ignore')

# 1. Load and preprocess data
df = pd.read_csv("data/driving_log.csv", header=None)

print(df.tail())

labels = df[3]
data = []

for i in range(len(df)):
    address = "data/" + df.iloc[i, 0]
    image = cv2.imread(address)

    #print(address)

    if image is None:
        continue

    # Crop road area
    cropped = image[60:135, :, :]

    # Convert to YUV
    yuv_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2YUV)

    # Resize
    resized = cv2.resize(yuv_image, (200, 66))

    # Gaussian Blur
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)

    # Normalize to float32 [0,1]
    normalized = blurred / 255.0
    normalized = normalized.astype('float32')

    data.append(normalized)

    if i % 200 == 0:
        print(f'[INFO] {i} images processed!')

X = np.array(data)
y = np.array(labels)

print(X.shape, y.shape)
print(y[:10])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Data augmentation functions

def augment_image(img, steering_angle):
    # Horizontal flip
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering_angle = -steering_angle

    # Brightness adjustment
    if np.random.rand() < 0.5:
        img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_YUV2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * ratio, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV) / 255.0

    # Zoom
    if np.random.rand() < 0.5:
        zoom = np.random.uniform(1.0, 1.2)
        h, w, _ = img.shape
        new_h, new_w = int(h / zoom), int(w / zoom)
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        img = img[top:top+new_h, left:left+new_w]
        img = cv2.resize(img, (w, h))

    # Pan
    if np.random.rand() < 0.5:
        h, w, _ = img.shape
        max_pan = 20
        tx = np.random.randint(-max_pan, max_pan)
        ty = np.random.randint(-max_pan, max_pan)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M, (w, h))

    # Rotation
    if np.random.rand() < 0.5:
        h, w, _ = img.shape
        angle = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))

    return img, steering_angle

# 3. Batch generators

def train_generator(X, y, batch_size):
    num_samples = len(X)
    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for offset in range(0, num_samples, batch_size):
            batch_idx = indices[offset:offset+batch_size]
            batch_images = []
            batch_angles = []
            for i in batch_idx:
                img, angle = augment_image(X[i], y[i])
                batch_images.append(img)
                batch_angles.append(angle)
            yield np.array(batch_images), np.array(batch_angles)
"""
def train_generator(X, y, batch_size):
    num_samples = len(X)
    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for offset in range(0, num_samples, batch_size):
            batch_idx = indices[offset:offset+batch_size]
            batch_images = []
            batch_angles = []
            for i in batch_idx:
                # No augmentation here — just original images and angles
                batch_images.append(X[i])
                batch_angles.append(y[i])
            yield np.array(batch_images), np.array(batch_angles)

"""
def val_generator(X, y, batch_size):
    num_samples = len(X)
    while True:
        for i in range(0, num_samples, batch_size):
            batch_images = X[i:i+batch_size]
            batch_angles = y[i:i+batch_size]
            yield batch_images, batch_angles

# 4. Model definition

model = Sequential([
    layers.Lambda(lambda x: x, input_shape=(66, 200, 3)),

    layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu'),
    layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
    layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(1164, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(50, activation='relu'),
    layers.Dense(10, activation='relu'),

    layers.Dense(1)
])

model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
model.summary()

# 5. Train the model using generators

batch_size = 64
steps_per_epoch = len(X_train) // batch_size
validation_steps = len(X_test) // batch_size

history = model.fit(
    train_generator(X_train, y_train, batch_size),
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator(X_test, y_test, batch_size),
    validation_steps=validation_steps,
    epochs=10
)

# 6. Plot training history

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Mean Absolute Error over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

plt.tight_layout()
plt.show()

# Save the trained model so TestSimulation.py can load it
model.save("Self_model.h5")
print("[INFO] Model saved as Self_model_2.h5")
