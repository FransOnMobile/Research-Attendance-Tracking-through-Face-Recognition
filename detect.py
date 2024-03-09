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
import json

def load_class_periods():
    global class_periods
    if os.path.exists(class_periods_file):
        with open(class_periods_file, 'r') as f:
            class_periods = json.load(f)

def get_current_class_period():
    now = datetime.now()
    day = now.strftime("%A")
    time = now.strftime("%H:%M")
    if day in class_periods:
        for period_start, period_end, period_name in class_periods[day]:
            if period_start <= time < period_end:
                return period_name, period_start, period_end
    return None, None, None

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
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
        attendance_tracker[name] = {"Status": {}, "Period": {}, "LastSeenTimestamp": None}
    
    # Get the current time
    current_time = datetime.now()
    
    # Check if the student was recognized recently
    last_seen_timestamp = attendance_tracker[name]["LastSeenTimestamp"]
    if last_seen_timestamp and (current_time - last_seen_timestamp).total_seconds() < 60:  # Adjust the time window as needed
        # Update the last seen timestamp
        timestamp_key = f"Timestamp{len(attendance_tracker[name]['Status'])}"
    else:
        # Add a new timestamp entry
        timestamp_key = f"Timestamp{len(attendance_tracker[name]['Status']) + 1}"
    
    # Update the attendance tracker
    attendance_tracker[name]["Status"][timestamp_key] = status
    attendance_tracker[name]["Period"][timestamp_key] = {"Name": period_name, "Start": period_start, "End": period_end}
    attendance_tracker[name]["LastSeenTimestamp"] = current_time
    
    # If the student is present, update the last seen timestamp for the student
    if status == "Present":
        attendance_tracker[name]["LastSeenTimestamp"] = current_time

def write_attendance_to_file(attendance_tracker, student_timestamps, timestamps, student_names, class_periods):
    logs_dir = "Attendance Logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Get the date for the attendance file
    date = timestamps[0].strftime('%Y-%m-%d')
    file_path = os.path.join(logs_dir, f"Attendance_{date}.xlsx")

    # Initialize DataFrame for attendance
    df_data = []

    # Add headers
    headers = ["DATE", "Name", "Period Name", "Period Start", "Period End", "Status"]
    max_timestamp_count = max(len(ts) for ts in student_timestamps.values()) if student_timestamps else 0
    headers += [f"Timestamp{i + 1}" for i in range(max_timestamp_count)]
    df_data.append(headers)

    # Write attendance data for each student
    for name in student_names:
        # Pad timestamps with empty strings to ensure equal length for all rows
        timestamps = [''] * max_timestamp_count
        period_info = [''] * 3  # for period name, start, and end
        status = "Absent"  # Default status
        if name in attendance_tracker:
            # Get the latest status from the attendance tracker
            status_data = attendance_tracker[name]["Status"]
            if status_data:
                last_key = max(status_data.keys(), key=lambda x: int(x.lstrip("Timestamp")))
                status = status_data[last_key]

            # Get period information if available
            period_data = attendance_tracker[name]["Period"]
            if period_data:
                last_key = max(period_data.keys(), key=lambda x: int(x.lstrip("Timestamp")))
                period_info = [period_data[last_key]["Name"], period_data[last_key]["Start"], period_data[last_key]["End"]]
                # Check if student is late based on period start time
                if status == "Absent" and last_key in status_data:
                    period_start = datetime.strptime(period_data[last_key]["Start"], "%H:%M")
                    last_seen_time = datetime.strptime(last_key, "%H:%M:%S")
                    if (last_seen_time - period_start).total_seconds() >= 600:  # 10 minutes in seconds
                        status = "Late"

            # Get recognition timestamps if available
            if name in student_timestamps:
                timestamps = [ts.strftime("%H:%M:%S") for ts in student_timestamps[name]]

        df_data.append([date, name] + period_info + [status] + timestamps)

    # Create DataFrame
    df = pd.DataFrame(df_data)

    # Write DataFrame to Excel file
    df.to_excel(file_path, index=False, header=False)

    # Print confirmation message
    print(f"Attendance file created: {file_path}")

def detect_faces(detector, img_rgb, results_queue):
    results = detector.detect_faces(img_rgb)
    results_queue.put(results)

def process_frame(frame, face_detector, face_encoder, encoding_dict, attendance_tracker, last_seen_faces, student_timestamps):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_queue = queue.Queue()
    detection_thread = threading.Thread(target=detect_faces, args=(face_detector, img_rgb, results_queue))
    detection_thread.start()
    detection_thread.join()
    results = results_queue.get()
    current_face_ids = set()
    current_time = datetime.now()
    for i, res in enumerate(results):
        if res['confidence'] < confidence_t:
            continue
        x1, y1, width, height = res['box']
        current_face_id = i + 1
        current_face_ids.add(current_face_id)
        face = frame[y1:y1+height, x1:x1+width]
        encode = get_encode(face_encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'
        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist <= recognition_t and dist < distance:
                name = db_name
                distance = dist
        period_name, period_start, period_end = get_current_class_period()
        if name != 'unknown':
            # Check if the last recognized timestamp for the student is different from the current time
            if name not in last_seen_faces or (current_time - last_seen_faces[name]).total_seconds() > 10:
                mark_attendance(attendance_tracker, name, "Present", current_time, period_name, period_start, period_end)
                last_seen_faces[name] = current_time
                if name not in student_timestamps:
                    student_timestamps[name] = []  # Initialize timestamps list for the student
                student_timestamps[name].append(current_time)  # Record timestamp of recognition for the student
        if name == 'unknown':
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), color, 2)
        cv2.putText(frame, f"{name}__{distance:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2, cv2.LINE_AA)
    return frame

def select_camera():
    devices = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                devices.append((i, cap.get(cv2.CAP_PROP_FOURCC)))
            cap.release()
    for i, (index, fourcc) in enumerate(devices):
        print(f"{i + 1}: Camera {index} - {fourcc_to_camera_name(fourcc)}")
    while True:
        choice = input("Choose a camera by entering its number: ")
        if choice.isdigit():
            choice = int(choice)
            if 1 <= choice <= len(devices):
                return devices[choice - 1][0]
        print("Invalid choice. Please enter a number corresponding to the camera.")

def fourcc_to_camera_name(fourcc):
    camera_names = {
        0x3CA8: "A4Tech Camera",
        0x4753: "GSou Camera"
    }
    return camera_names.get(int(fourcc), "Unknown")

if __name__ == "__main__":
    confidence_t = 0.95
    recognition_t = 0.3
    required_size = (160, 160)

    class_periods_file = 'class_periods.json'
    class_periods = {}

    face_encoder = InceptionResNetV2()
    path_m = "facenet_keras_weights.h5" 
    try:
        face_encoder.load_weights(path_m)
    except FileNotFoundError:
        print(f"Error: Model weights file '{path_m}' not found.")
        exit(1)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        exit(1)
    encodings_path = 'encodings/encodings.pkl'
    try:
        encoding_dict = load_pickle(encodings_path)
    except FileNotFoundError:
        print(f"Error: Encoding file '{encodings_path}' not found.")
        exit(1)
    except Exception as e:
        print(f"Error loading encoding dictionary: {e}")
        exit(1)
    face_detector = mtcnn.MTCNN()
    encoding_dict = load_pickle(encodings_path)
    student_names = [folder for folder in os.listdir('Faces') if os.path.isdir(os.path.join('Faces', folder))]

    attendance_tracker = {}
    last_seen_faces = {}
    start_time = datetime.now()
    timestamps = [start_time]
    load_class_periods()
    selected_camera = select_camera()
    student_timestamps = {}

    cap = cv2.VideoCapture(selected_camera)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to retrieve frame from camera.")
            break
        current_time = datetime.now()
        if (current_time - timestamps[-1]).total_seconds() >= 600:
            timestamps.append(current_time)
        current_period = get_current_class_period()
        if current_period[0] is not None and "snack" not in current_period[0].lower():
            frame = process_frame(frame, face_detector, face_encoder, encoding_dict, attendance_tracker, last_seen_faces, student_timestamps)

        cv2.imshow('camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

write_attendance_to_file(attendance_tracker, student_timestamps, timestamps, student_names)
