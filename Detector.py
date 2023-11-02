import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

# Load pre-trained face detection and recognition models
face_detector = MTCNN()
face_recognizer = InceptionResnetV1(pretrained='vggface2').eval()

def main_app(name):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        faces = face_detector(frame)

        if faces is not None and len(faces) > 0:
            for face in faces:
                # Check the structure of the 'box' variable
                if isinstance(face, dict) and 'box' in face:
                    box = face['box']  # Get the bounding box information
                    x, y, w, h = box  # Unpack the box coordinates

                    # Extract and preprocess the detected face
                    face_img = frame[y:y+h, x:x+w]  # Use the coordinates to crop the face

                    # Check if the extracted face is valid
                    if not face_img.size == 0:
                        # Convert to RGB if not already in RGB
                        if face_img.shape[-1] == 1:
                            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
                        elif face_img.shape[-1] == 3 and not cv2.COLOR_BGR2RGB:
                            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                        # Resize and normalize
                        face_img = cv2.resize(face_img, (160, 160))
                        face_img = (face_img / 255.0 - 0.5) * 2.0  # Normalize to [-1, 1]

                        # Convert to PyTorch tensor
                        face_tensor = torch.from_numpy(face_img.transpose((2, 0, 1))).unsqueeze(0).float()

                        # Compute embeddings
                        embeddings = face_recognizer(face_tensor)

                        # Perform recognition and display results
                        # Compare embeddings with known embeddings and determine if it's a match
                        # Display recognized name and confidence score

        cv2.imshow("image", frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
