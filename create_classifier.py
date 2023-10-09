import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Define a CNN model for face recognition
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

# Load the dataset (assumes each subfolder in 'data' contains images of one person)
def load_dataset(data_dir, target_size=(224, 224)):
    images = []
    labels = []

    for subfolder in os.listdir(data_dir):
        subfolder_path = os.path.join(data_dir, subfolder)
        if os.path.isdir(subfolder_path):
            label = subfolder
            for image_file in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder_path, image_file)
                image = cv2.imread(image_path)
                # Resize the image to the target size
                image = cv2.resize(image, target_size)
                images.append(image)
                labels.append(label)

    return np.array(images), np.array(labels)

# Main function
def train_classifier(data_directory, model_save_path):
    print("data_directory =", data_directory)

    # Check if the data directory exists
    if not os.path.exists(data_directory):
        print(f"The 'data' folder does not exist at: {data_directory}")
    else:
        # List subfolders in the data directory
        subfolders = [f.name for f in os.scandir(data_directory) if f.is_dir()]

        if len(subfolders) == 0:
            print("No subfolders found in 'data' directory.")
        else:
            print("Subfolders in 'data' directory:")
            for subfolder in subfolders:
                print(subfolder)


    # Load the dataset
    images, labels = load_dataset(data_directory)

    # Encode labels as integers
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

    # Convert labels to one-hot encoding
    num_classes = len(label_encoder.classes_)
    y_train_onehot = to_categorical(y_train, num_classes=num_classes)
    y_test_onehot = to_categorical(y_test, num_classes=num_classes)

    # Preprocess images (normalize pixel values to [0, 1])
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Create and compile the CNN model
    input_shape = X_train[0].shape
    model = create_cnn_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train_onehot, epochs=10, batch_size=32, validation_data=(X_test, y_test_onehot))

    # Save the trained model
    model.save(model_save_path)

if __name__ == '__main__':
    data_directory = os.path.join(os.getcwd(), "data")
    print("data_directory =", data_directory)
    print("Subfolders in data_directory:", os.listdir(data_directory))
    save_path = 'face_recognition_model.h5'
    train_classifier(data_directory, save_path)
