import os
import cv2
import numpy as np
import mtcnn
import pickle
from architecture import *
from train_v2 import normalize, l2_normalizer
from scipy.spatial.distance import cosine
from datetime import datetime, timedelta
import pandas as pd
import threading
import queue
import json  # Import the json module

confidence_t = 0.99
recognition_t = 0.5
required_size = (160, 160)
zoomed_face_window_open = False
face_id_counter = 0
absent_duration_threshold = timedelta(minutes=50)  # Adjust as needed

# Load class periods from JSON file
class_periods_file = 'class_periods.json'
if os.path.exists(class_periods_file):
    with open(class_periods_file, 'r') as f:
        class_periods = json.load(f)
else:
    class_periods = {}  # Create an empty dictionary if the file doesn't exist

def get_current_class_period():
    # Get the current day and time
    now = datetime.now()
    day = now.strftime("%A")
    time = now.strftime("%H:%M")

    # Check if the current time falls within any class period
    if day in class_periods:
        for period_start, period_end, period_name in class_periods[day]:
            if period_start <= time < period_end:
                return period_name, period_start, period_end

    return None, None, None

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

def mark_attendance(attendance_tracker, name, status, timestamp, period_name, period_start, period_end):
    if name not in attendance_tracker:
        attendance_tracker[name] = {"Status": {}, "Period": {}}

    attendance_tracker[name]["Status"][timestamp] = status
    attendance_tracker[name]["Period"][timestamp] = {"Name": period_name, "Start": period_start, "End": period_end}

def write_attendance_to_file(attendance_tracker, timestamps):
    # Create the "Attendance Logs" directory if it doesn't exist
    logs_dir = "Attendance Logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Write attendance data for each day to a separate Excel file
    for timestamp in timestamps:
        # Get the date
        date = timestamp.date().strftime('%Y-%m-%d')

        # Create a DataFrame for the attendance data
        df_data = []
        for name, data in attendance_tracker.items():
            for ts, status in data["Status"].items():
                period_info = data["Period"][ts]
                df_data.append([ts, name, status, period_info["Name"], period_info["Start"], period_info["End"]])
        df = pd.DataFrame(df_data, columns=["Timestamp", "Name", "Status", "Period Name", "Period Start", "Period End"])

        # Write DataFrame to Excel file
        file_path = os.path.join(logs_dir, f"Attendance_{date}.xlsx")
        df.to_excel(file_path, index=False)

def detect_faces(detector, img_rgb, results_queue):
    results = detector.detect_faces(img_rgb)
    results_queue.put(results)

def process_frame(frame, face_detector, face_encoder, encoding_dict, attendance_tracker, last_seen_faces):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_queue = queue.Queue()

    # Perform face detection in a separate thread
    detection_thread = threading.Thread(target=detect_faces, args=(face_detector, img_rgb, results_queue))
    detection_thread.start()
    detection_thread.join()

    results = results_queue.get()

    # Create a set to keep track of the current face IDs
    current_face_ids = set()

    for i, res in enumerate(results):  # Use enumerate to loop over the results
        if res['confidence'] < confidence_t:
            continue

        # Original face region coordinates
        x1, y1, width, height = res['box']

        # Generate a unique face ID using the loop index
        current_face_id = i + 1
        current_face_ids.add(current_face_id)

        face = frame[y1:y1+height, x1:x1+width]
        encode = get_encode(face_encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        print(f"Detected: {name} (Confidence: {res['confidence']:.2f})")

        # Get current class period
        period_name, period_start, period_end = get_current_class_period()

        if name != 'unknown':
            mark_attendance(attendance_tracker, name, "Present", datetime.now(), period_name, period_start, period_end)
            last_seen_faces[name] = datetime.now()

        if name == 'unknown':
            cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (0, 0, 255), 2)
            cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)  # Use cv2.LINE_AA for smoother text
        else:
            cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 2)
            cv2.putText(frame, f"{name}__{distance:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2, cv2.LINE_AA)  # Use f-strings to format the text

    # Mark absent for faces not seen for a certain duration
    for name, last_seen_time in last_seen_faces.items():
        if (datetime.now() - last_seen_time) >= absent_duration_threshold:
            # Get current class period
            period_name, period_start, period_end = get_current_class_period()
            mark_attendance(attendance_tracker, name, "Absent", datetime.now(), period_name, period_start, period_end)

    return frame

def select_camera():
    # Get available camera devices
    devices = []
    for i in range(10):  # Check up to 10 cameras
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                devices.append((i, cap.get(cv2.CAP_PROP_FOURCC)))
            cap.release()

    # Print available camera devices
    for i, (index, fourcc) in enumerate(devices):
        print(f"{i + 1}: Camera {index} - {fourcc_to_camera_name(fourcc)}")

    # Ask user to choose a camera
    while True:
        choice = input("Choose a camera by entering its number: ")
        if choice.isdigit():
            choice = int(choice)
            if 1 <= choice <= len(devices):
                return devices[choice - 1][0]
        print("Invalid choice. Please enter a number corresponding to the camera.")

def fourcc_to_camera_name(fourcc):
    # You may need to adjust this dictionary based on the actual names of your cameras
    camera_names = {
        0x3CA8: "A4Tech Camera",
        0x4753: "GSou Camera"
        # Add more mappings as necessary
    }
    return camera_names.get(int(fourcc), "Unknown")

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
    last_seen_faces = {}  # Store the timestamp when each face was last seen

    # Initialize variables for time tracking
    start_time = datetime.now()

    timestamps = [start_time]

    # Select camera
    selected_camera = select_camera()

    cap = cv2.VideoCapture(selected_camera)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("CAM NOT OPENED")
            break

        current_time = datetime.now()

        # Check if 10 minutes have elapsed since the last timestamp
        if (current_time - timestamps[-1]).total_seconds() >= 600:
            timestamps.append(current_time)

        # Get current class period
        current_period = get_current_class_period()

        # Process frame only during class periods and not during snack breaks
        if current_period[0] is not None and "snack" not in current_period[0].lower():
            # Process frame asynchronously
            frame = process_frame(frame, face_detector, face_encoder, encoding_dict, attendance_tracker, last_seen_faces)

        cv2.imshow('camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Write attendance data to file
    write_attendance_to_file(attendance_tracker, timestamps)
