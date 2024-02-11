import cv2
import os
import numpy as np
import mtcnn
import pickle
from architecture import *
from train_v2 import normalize, l2_normalizer
from scipy.spatial.distance import cosine
from datetime import datetime, timedelta
import pandas as pd

confidence_t = 0.99
recognition_t = 0.5
required_size = (160, 160)
zoomed_face_window_open = False
face_id_counter = 0

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    # Crop the face without explicit zoom factor
    face = img[y1:y2, x1:x2]

    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode

def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def mark_attendance(attendance_tracker, name, status, timestamp):
    if name not in attendance_tracker:
        attendance_tracker[name] = {}

    attendance_tracker[name][timestamp] = status

def write_attendance_to_file(attendance_tracker, timestamps):
    with open('attendance_record.csv', 'w') as f:
        # Write header
        f.write('Timestamp,' + ','.join(attendance_tracker.keys()) + '\n')

        # Write attendance data for each timestamp
        for timestamp in timestamps:
            line = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            for name, status_dict in attendance_tracker.items():
                status = status_dict.get(timestamp, 'Absent')
                line += ',' + status
            f.write(line + '\n')

def detect(img, detector, encoder, encoding_dict, attendance_tracker, min_zoom_factor=1.0, max_zoom_factor=10.0):
    global face_id_counter, zoomed_face_window_open
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)

    # Close the 'Zoomed Face' window if no faces are detected
    if not results and zoomed_face_window_open:
        cv2.destroyWindow('Zoomed Face')
        zoomed_face_window_open = False
        return img

    # Create a set to keep track of the current face IDs
    current_face_ids = set()

    for i, res in enumerate(results):  # Use enumerate to loop over the results
        if res['confidence'] < confidence_t:
            continue

        # Original face region coordinates
        x1, y1, width, height = res['box']

        # Calculate distance of the face from the camera using the bounding box size
        face_size = np.sqrt(width * height)
        distance_from_camera = face_size  # Directly use the size of the bounding box

        # Define a target face size (e.g., 160x160 pixels)
        target_face_size = (160, 160)

        # Calculate the zoom factor based on the target face size and the distance from the camera
        zoom_factor = np.interp(distance_from_camera, [10, 50], [max_zoom_factor, min_zoom_factor])

        # Ensure the zoom factor is within the specified range
        zoom_factor = max(min_zoom_factor, min(max_zoom_factor, zoom_factor))

        # Calculate the center of the face
        face_center_x = x1 + width // 2
        face_center_y = y1 + height // 2

        # Calculate the zoomed-in face region size
        zoomed_width = int(width * zoom_factor)
        zoomed_height = int(height * zoom_factor)

        # Crop and resize the zoomed-in face region using cv2.getRectSubPix
        zoomed_face = cv2.getRectSubPix(img_rgb, (zoomed_width, zoomed_height), (face_center_x, face_center_y))
        zoomed_face = cv2.resize(zoomed_face, target_face_size)

        # Display the zoomed-in face in the window
        cv2.imshow('Zoomed Face', cv2.cvtColor(zoomed_face, cv2.COLOR_RGB2BGR))

        # Generate a unique face ID using the loop index
        current_face_id = i + 1
        current_face_ids.add(current_face_id)

        encode = get_encode(encoder, zoomed_face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        print(f"Detected: {name} (Confidence: {res['confidence']:.2f}, Distance: {distance_from_camera:.2f}, Zoom Factor: {zoom_factor:.2f})")

        if name != 'unknown':
            mark_attendance(attendance_tracker, name, "Present", datetime.now())

        if name == 'unknown':
            cv2.rectangle(img, (x1, y1), (x1 + width, y1 + height), (0, 0, 255), 2)
            cv2.putText(img, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)  # Use cv2.LINE_AA for smoother text
        else:
            cv2.rectangle(img, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 2)
            cv2.putText(img, f"{name}__{distance:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2, cv2.LINE_AA)  # Use f-strings to format the text

    return img


if __name__ == "__main__":
    required_shape = (160, 160)
    face_encoder = InceptionResNetV2()
    path_m = "facenet_keras_weights.h5"
    face_encoder.load_weights(path_m)
    encodings_path = 'encodings/encodings.pkl'
    face_detector = mtcnn.MTCNN()
    encoding_dict = load_pickle(encodings_path)
    student_names = [folder for folder in os.listdir('Faces') if os.path.isdir(os.path.join('Faces', folder))]

    # Initialize dictionary to store attendance data
    attendance_tracker = {}

    # Initialize variables for time tracking
    start_time = datetime.now()

    timestamps = [start_time]

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("CAM NOT OPENED")
            break

        current_time = datetime.now()

        # Check if 10 minutes have elapsed since the last timestamp
        if (current_time - timestamps[-1]).total_seconds() >= 600:
            timestamps.append(current_time)

        frame = detect(frame, face_detector, face_encoder, encoding_dict, attendance_tracker)

        cv2.imshow('camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Write attendance data to file
    write_attendance_to_file(attendance_tracker, timestamps)
