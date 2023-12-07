from architecture import * 
import os 
import cv2
import mtcnn
import pickle 
import numpy as np 
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import load_model
from tqdm import tqdm

###### paths and variables#########
face_data = 'Faces/'
required_shape = (160, 160)
face_encoder = InceptionResNetV2()
path = "facenet_keras_weights.h5"
face_encoder.load_weights(path)
face_detector = mtcnn.MTCNN()
encodes = []
encoding_dict = dict()
l2_normalizer = Normalizer('l2')
total_images = sum(len(os.listdir(os.path.join(face_data, face_names))) for face_names in os.listdir(face_data))
print(f"Total images for training: {total_images}")
###############################

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

for face_names in os.listdir(face_data):
    person_dir = os.path.join(face_data, face_names)
    
    print(f"Training {face_names}...")
    for image_name in tqdm(os.listdir(person_dir), desc=f'Training {face_names}', total=total_images, position=0, leave=True):
        image_path = os.path.join(person_dir, image_name)

        img_BGR = cv2.imread(image_path)
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

        x = face_detector.detect_faces(img_RGB)
        if x:
            x1, y1, width, height = x[0]['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = img_RGB[y1:y2, x1:x2]

            face = normalize(face)
            face = cv2.resize(face, required_shape)
            face_d = np.expand_dims(face, axis=0)
            encode = face_encoder.predict(face_d)[0]
            encodes.append(encode)
        else:
            print("No faces detected in the image.")

        face = normalize(face)
        face = cv2.resize(face, required_shape)
        face_d = np.expand_dims(face, axis=0)
        encode = face_encoder.predict(face_d)[0]
        encodes.append(encode)

    print(f"Training for {face_names} completed.")
    if encodes:
        encode = np.sum(encodes, axis=0)
        encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
        encoding_dict[face_names] = encode

# Display the list of names trained
print("List of names trained:")
for name in encoding_dict:
    print(name)

path = 'encodings/encodings.pkl'
with open(path, 'wb') as file:
    pickle.dump(encoding_dict, file)

print("Training completed.")
