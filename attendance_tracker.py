import csv
import os
from datetime import datetime, timedelta

class AttendanceTracker:
    def __init__(self, student_names, output_file="attendance.csv"):
        self.student_names = student_names
        self.output_file = output_file
        self.attendance_data = {name: "Absent" for name in student_names}
        self.last_seen = {name: None for name in student_names}

        # Load existing attendance data if available
        self.load_attendance_data()

    def load_attendance_data(self):
        if not os.path.exists(self.output_file):
            # If the CSV file doesn't exist, create it with headers
            with open(self.output_file, 'w', newline='') as csvfile:
                fieldnames = ['StudentName', 'Status', 'Timestamp']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        with open(self.output_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                student_name = row['StudentName']
                status = row['Status']
                timestamp = row['Timestamp']

                self.attendance_data[student_name] = status
                self.last_seen[student_name] = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')

    def mark_attendance(self, student_name, status):
        timestamp = datetime.now()
        self.attendance_data[student_name] = status
        self.last_seen[student_name] = timestamp

    def write_attendance_to_csv(self):
        print("Write attendance is being called")
        with open(self.output_file, 'a', newline='') as csvfile:
            fieldnames = ['StudentName', 'Status', 'Timestamp']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if csvfile.tell() == 0:
                writer.writeheader()

            current_time = datetime.now()

            for student_name, last_seen_time in self.last_seen.items():
                if last_seen_time is None or current_time - last_seen_time > timedelta(minutes=10):
                    status = "Absent"
                else:
                    status = self.attendance_data[student_name]

                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                writer.writerow({'StudentName': student_name, 'Status': status, 'Timestamp': timestamp})

    def get_attendance_data(self):
        return self.attendance_data

    def update_attendance_status(self):
        current_time = datetime.now()

        for student_name, last_seen_time in self.last_seen.items():
            if last_seen_time is None:
                self.attendance_data[student_name] = "Absent"
            elif current_time - last_seen_time <= timedelta(minutes=10):
                self.attendance_data[student_name] = "Present"
            else:
                self.attendance_data[student_name] = "Cutting Classes"
