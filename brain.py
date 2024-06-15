import os
import cv2
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.api.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
# Function to load brain tumor images from a folder
def load_brain_tumor_images_from_folder(folder, label, subset_size):
    images = []
    labels = []
    for filename in os.listdir(folder)[:subset_size]:
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (224, 224))
            images.append(img)
            labels.append(label)
    return images, labels

# Set the number of samples per class for the subset (limited dataset)
subset_size = 101

# Load brain tumor and normal brain images
tumor_images, tumor_labels = load_brain_tumor_images_from_folder(r"C:\Users\paruc\Downloads\brain tumour\Brain Tumor Data Set\Brain Tumor Data Set\Healthy", 1, subset_size)
normal_images, normal_labels = load_brain_tumor_images_from_folder(r"C:\Users\paruc\Downloads\brain tumour\Brain Tumor Data Set\Brain Tumor Data Set\Brain Tumor", 0, subset_size)

# Combine tumor and normal images and labels
images = tumor_images[:subset_size] + normal_images[:subset_size]
labels = tumor_labels[:subset_size] + normal_labels[:subset_size]

# Convert images and labels to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Normalize images
images = images / 255.0

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top of the pre-trained base model
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=16)

# Evaluate the model on the training set
loss, accuracy = model.evaluate(X_train, y_train)
print('Training Loss:', loss)
print('Training Accuracy:', accuracy)

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(X_val, y_val)
print('Validation Loss:', loss)
print('Validation Accuracy:', accuracy)

# Save the model to a file
model.save("brain_tumor_model_vgg16_with_augmentation.h5")