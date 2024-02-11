import cv2, os
import numpy as np
import mtcnn
from architecture import *
from train_v2 import normalize, l2_normalizer
from scipy.spatial.distance import cosine
from tensorflow.keras.models import load_model
from attendance_tracker import AttendanceTracker
import pickle
import time
from datetime import datetime, timedelta

confidence_t = 0.99
recognition_t = 0.5
required_size = (160, 160)
zoomed_face_window_open = False
face_id_counter=0

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

import cv2
import numpy as np

# Initialize a counter for generating unique face IDs
face_id_counter = 0

# Initialize the window name for the zoomed face
zoomed_face_window_name = 'Zoomed Face'

# ... (previous code remains unchanged)

def detect(img, detector, encoder, encoding_dict, attendance_tracker, min_zoom_factor=1.0, max_zoom_factor=10.0):
    global face_id_counter, zoomed_face_window_open  # Use global variables

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)

    present_students = set()

    # Close the 'Zoomed Face' window if no faces are detected
    if not results and zoomed_face_window_open:
        cv2.destroyWindow(zoomed_face_window_name)
        zoomed_face_window_open = False
        return img

    # Create a set to keep track of the current face IDs
    current_face_ids = set()

    for res in results:
        if res['confidence'] < confidence_t:
            continue

        # Original face region coordinates
        x1, y1, width, height = res['box']

        # Calculate distance of the face from the camera using the bounding box size
        face_size = np.sqrt(width * height)
        distance_from_camera = face_size  # Directly use the size of the bounding box

        # Use an exponential function for dynamic zooming
        dynamic_zoom_factor = np.interp(distance_from_camera, [10, 50], [max_zoom_factor, min_zoom_factor])

        print(f"Debug - Distance: {distance_from_camera:.2f}, Zoom Factor: {dynamic_zoom_factor:.2f}")

        # Calculate zoomed-in coordinates
        zoom_x1 = max(0, int(x1 - width * (dynamic_zoom_factor - 1) / 2))
        zoom_y1 = max(0, int(y1 - height * (dynamic_zoom_factor - 1) / 2))
        zoom_x2 = min(img.shape[1], int(x1 + width * (dynamic_zoom_factor - 1) / 2 + width))
        zoom_y2 = min(img.shape[0], int(y1 + height * (dynamic_zoom_factor - 1) / 2 + height))

        # Crop and zoom the face
        face = img_rgb[zoom_y1:zoom_y2, zoom_x1:zoom_x2]

        # Display the zoomed-in face in the window
        cv2.imshow(zoomed_face_window_name, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

        # Generate a unique face ID
        face_id_counter += 1
        current_face_id = face_id_counter
        current_face_ids.add(current_face_id)

        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        print(f"Detected: {name} (Confidence: {res['confidence']:.2f}, Distance: {distance_from_camera:.2f}, Zoom Factor: {dynamic_zoom_factor:.2f})")

        attendance_tracker.mark_attendance(name, "Present")
        present_students.add(name)

        if name == 'unknown':
            cv2.rectangle(img, (x1, y1), (x1 + width, y1 + height), (0, 0, 255), 2)
            cv2.putText(img, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 2)
            cv2.putText(img, f"{name}__{distance:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)

        current_time = datetime.now().timestamp()

        for student_name, last_seen_time in attendance_tracker.last_seen.items():
            if (
                student_name not in present_students
                and last_seen_time is not None
                and current_time - last_seen_time.timestamp() <= 600
            ):
                attendance_tracker.mark_attendance(student_name, "Absent")

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
    attendance_tracker = AttendanceTracker(student_names)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("CAM NOT OPENED")
            break

        frame = detect(frame, face_detector, face_encoder, encoding_dict, attendance_tracker)

        cv2.imshow('camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    attendance_tracker.update_attendance_status()
    attendance_tracker.write_attendance_to_csv()
    print("Final Attendance Data:")
    for student_name, attendance_status in attendance_tracker.get_attendance_data().items():
        print(f"{student_name}: {attendance_status}")
