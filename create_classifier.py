import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Function to create the CNN model
def create_cnn_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Load the dataset
def load_dataset(data_dir, target_size=(160, 160)):
    images = []
    labels = []

    for subfolder in os.listdir(data_dir):
        subfolder_path = os.path.join(data_dir, subfolder)
        if os.path.isdir(subfolder_path):
            label = subfolder
            for image_file in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder_path, image_file)
                image = cv2.imread(image_path)
                if image is not None and image.size > 0:
                    resized_image = cv2.resize(image, target_size)
                    images.append(resized_image)
                    labels.append(label)
    
    return np.array(images), np.array(labels)

# Train the classifier
def train_classifier(data_directory):
    images, labels = load_dataset(data_directory)
    
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    y = to_categorical(labels_encoded, num_classes=num_classes)
    
    X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.2, random_state=42)
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    input_shape = X_train[0].shape
    model_path = os.path.join(data_directory, "consolidated_model.keras")
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
    else:
        model = create_cnn_model(input_shape, num_classes)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Retrain the model on new faces using the LwF technique
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
    
    # Save the consolidated model after training
    model.save(model_path)