import os, cv2, re, threading
import shutil, csv, datetime, torch, threading
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
from tkinter import messagebox
from facenet_pytorch import MTCNN, InceptionResnetV1
from Detector import main_app
from create_classifier import train_classifier
from create_dataset import start_capture

names = set()


class MainUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.clear_data_button = None
        global names
        with open("nameslist.txt", "r") as f:
            x = f.read()
            z = x.rstrip().split(" ")
            for i in z:
                names.add(i)
        self.title_font = tkfont.Font(family='Helvetica', size=16, weight="bold")
        self.title("Attendance Tracking Research Project")
        self.resizable(True, True)
        self.geometry("730x600")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.active_name = None
        self.num_of_images = 0  # Initialize num_of_images
        self.ensure_attendance_log_exists()
        container = tk.Frame(self)
        container.grid(sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        self.frames["StartPage"] = StartPage(parent=container, controller=self)
        self.frames["MultiFaceDetectionPage"] = MultiFaceDetectionPage(parent=container, controller=self)
        for F in (StartPage, PageOne, PageTwo, PageThree, PageFour, AttendanceLogPage, MultiFaceDetectionPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame("StartPage")

    def load_names_from_file(self):
        with open("nameslist.txt", "r") as f:
            x = f.read()
            z = x.rstrip().split(" ")
            for i in z:
                self.names.add(i)

    def refresh_student_list(self):
        self.frames["StartPage"].refresh_student_list()

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

    def on_closing(self):
        if messagebox.askokcancel("Close", "Are you sure?"):
            global names
            f = open("nameslist.txt", "a+")
            for i in names:
                f.write(i + " ")
            self.destroy()

    def ensure_attendance_log_exists(self):
        log_file = "attendance_log.csv"
        if not os.path.exists(log_file):
            with open(log_file, "w", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Student Name"])


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.student_list_var = tk.StringVar()

        title_label = tk.Label(self, text="Attendance Tracking System", font=("Helvetica", 24), fg="#263942")
        title_label.pack(pady=20)
        
        student_list_frame = tk.Frame(self, padx=20, pady=20)
        student_list_frame.pack()
        
        # Header for the list
        list_header = tk.Label(student_list_frame, text="Registered Students", font=("Helvetica", 16), fg="#263942")
        list_header.grid(row=0, column=0, columnspan=2, pady=10)

        # frame for buttons
        button_frame = tk.Frame(self, padx=20, pady=20)
        button_frame.pack()
        
        # Listbox to display the students
        self.student_listbox = tk.Listbox(student_list_frame, listvariable=self.student_list_var, selectmode=tk.SINGLE, width=40, height=10)
        self.student_listbox.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        # buttons
        add_user_button = tk.Button(button_frame, text="Add a Student", fg="white", bg="#263942",
                                    command=lambda: controller.show_frame("PageOne"), padx=20, pady=10, font=("Helvetica", 14))
        check_user_button = tk.Button(button_frame, text="Check a Student", fg="white", bg="#263942",
                                      command=lambda: controller.show_frame("PageTwo"), padx=20, pady=10, font=("Helvetica", 14))
        clear_data_button = tk.Button(button_frame, text="Clear Data", fg="#ffffff", bg="#FF0000",
                                      command=self.confirm_clear_data, padx=20, pady=10, font=("Helvetica", 14))
        quit_button = tk.Button(button_frame, text="Quit", fg="#263942", bg="white", command=self.on_closing,
                                padx=20, pady=10, font=("Helvetica", 14))

        add_user_button.pack(side=tk.LEFT, padx=10)
        check_user_button.pack(side=tk.LEFT, padx=10)
        clear_data_button.pack(side=tk.LEFT, padx=10)
        quit_button.pack(side=tk.LEFT, padx=10)

        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1) 
        button_frame.columnconfigure(2, weight=1)
        button_frame.columnconfigure(3, weight=1)
        student_list_frame.columnconfigure(0, weight=1)
        student_list_frame.columnconfigure(1, weight=1)
        
        # Populate the listbox with sorted student names
        self.refresh_student_list()
        
        multi_face_detection_button = tk.Button(self, text="Multiple Face Detection",
                                                command=self.show_multi_face_detection, 
                                                fg="#ffffff", bg="#263942")
        multi_face_detection_button.pack(ipady=3, ipadx=7, padx=10, pady=10)


    def on_closing(self):
        if messagebox.askokcancel("Quit", "Are you sure?"):
            global names
            with open("nameslist.txt", "w") as f:
                for i in names:
                    f.write(i + " ")
            self.controller.destroy()
    
    def show_multi_face_detection(self):
        print("Showing MultiFaceDetectionPage")
        self.controller.show_frame("MultiFaceDetectionPage")
          
    def confirm_clear_data(self):
        if messagebox.askyesno("Clear Data", "Are you sure you want to clear all data?"):
            self.clear_data()

    def clear_data(self):
        global names
        names.clear()
        for name in os.listdir('./data'):
            if os.path.isdir(os.path.join('./data', name)):
                shutil.rmtree(os.path.join('./data', name))

        open("nameslist.txt", "w").close()
        self.controller.show_frame("StartPage")
        print("Cleared Data!")
        
    def refresh_student_list(self):
        self.student_listbox.delete(0, tk.END)

        # Update the listbox with sorted student names
        sorted_names = sorted(names)
        for name in sorted_names:
            self.student_listbox.insert(tk.END, name)


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        tk.Label(self, text="Enter the name of the student", fg="#263942", font='Helvetica 12 bold').grid(row=0, column=0, pady=10, padx=5)

        self.user_name = tk.Entry(self, borderwidth=3, bg="lightgrey", font='Helvetica 11')
        self.user_name.grid(row=0, column=1, pady=10, padx=10)

        self.buttoncanc = tk.Button(self, text="Cancel", bg="#ffffff", fg="#263942", command=lambda: controller.show_frame("StartPage"))
        self.buttonext = tk.Button(self, text="Next", fg="#ffffff", bg="#263942", command=self.start_training)
        self.buttoncanc.grid(row=1, column=0, pady=10, ipadx=5, ipady=4)
        self.buttonext.grid(row=1, column=1, pady=10, ipadx=5, ipady=4)

    def start_training(self):
        global names
        if self.user_name.get() == "None":
            messagebox.showerror("Error", "Name cannot be 'None'")
            return
        elif self.user_name.get() in names:
            messagebox.showerror("Error", "This student already exists!")
            return
        elif len(self.user_name.get()) == 0:
            messagebox.showerror("Error", "Name cannot be empty!")
            return

        name = self.user_name.get()
        names.add(name)
        with open("nameslist.txt", "a+") as f:
            f.write(name + " ")

        self.controller.active_name = name
        self.controller.frames["PageTwo"].refresh_names()
        self.controller.show_frame("PageThree")


class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.names = names
        self.controller = controller
        tk.Label(self, text="Select user", fg="#263942", font='Helvetica 12 bold').grid(row=0, column=0, padx=10,
                                                                                           pady=10)
        self.buttoncanc = tk.Button(self, text="Cancel", command=lambda: controller.show_frame("StartPage"),
                                    bg="#ffffff", fg="#263942")
        self.menuvar = tk.StringVar(self)
        self.dropdown = tk.OptionMenu(self, self.menuvar, *names)
        self.dropdown.config(bg="lightgrey")
        self.dropdown["menu"].config(bg="lightgrey")
        self.buttonext = tk.Button(self, text="Next", command=self.nextfoo, fg="#ffffff", bg="#263942")
        self.dropdown.grid(row=0, column=1, ipadx=8, padx=10, pady=10)
        self.buttoncanc.grid(row=1, ipadx=5, ipady=4, column=0, pady=10)
        self.buttonext.grid(row=1, ipadx=5, ipady=4, column=1, pady=10)

    def nextfoo(self):
        if self.menuvar.get() == "None":
            messagebox.showerror("ERROR", "Name cannot be 'None'")
            return
        self.controller.active_name = self.menuvar.get()
        self.controller.show_frame("PageFour")

    def refresh_names(self):
        self.menuvar.set('')
        self.dropdown['menu'].delete(0, 'end')
        for name in self.names:
            self.dropdown['menu'].add_command(label=name, command=tk._setit(self.menuvar, name))


def list_available_cameras():
    available_cameras = []

    # Enumerate camera devices and retrieve their names
    for i in range(10):  # Check up to 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            camera_name = get_camera_name(i)
            available_cameras.append((i, camera_name))
            cap.release()

    return available_cameras

def get_camera_name(camera_index):
    try:
        # Use OpenCV to get camera name on Windows
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            description = cap.getBackendName()

            name_mapping = {
                "MSMF": "Front Camera",
                "DSHOW": "USB Camera",
                # Add more mappings as needed
            }
            camera_name = name_mapping.get(description, description)
            return camera_name
        
        if not cap.isOpened():
            print("Error: Could not open camera.")


    except Exception as e:
        print(f"Error getting camera name: {str(e)}")

    return f"Camera {camera_index}"  # Fallback to default naming

def load_face_recognition_model():
    face_recognizer = InceptionResnetV1(pretrained='vggface2').eval()
    return face_recognizer

class PageThree(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.numimglabel = tk.Label(self, text="Number of images captured = 0", font='Helvetica 12 bold', fg="#263942")
        self.numimglabel.grid(row=0, column=0, columnspan=2, sticky="ew", pady=10)
        self.capturebutton = tk.Button(self, text="Capture Data Set", fg="#ffffff", bg="#263942", command=self.capimg)
        self.trainbutton = tk.Button(self, text="Train The Model", fg="#ffffff", bg="#263942", command=self.trainmodel)
        self.capturebutton.grid(row=1, column=0, ipadx=5, ipady=4, padx=10, pady=20)
        self.trainbutton.grid(row=1, column=1, ipadx=5, ipady=4, padx=10, pady=20)
        
        # dropdown menu for camera selection
        self.camera_label = tk.Label(self, text="Select Camera:", font='Helvetica 12 bold', fg="#263942")
        self.available_cameras = list_available_cameras()
        camera_names = [name for (_, name) in self.available_cameras]
        self.camera_var = tk.StringVar(self)
        self.camera_dropdown = ttk.Combobox(self, textvariable=self.camera_var, state="readonly")
        self.camera_dropdown['values'] = camera_names
        self.camera_dropdown.grid(row=2, column=0, columnspan=2, pady=10)
        
        self.camera_index_name_map = {name: index for index, name in self.available_cameras}

    def capimg(self):
        selected_camera_name = self.camera_var.get()
        if not selected_camera_name:
            messagebox.showerror("Error", "Please select a camera.")
            return

        selected_camera_index = self.camera_index_name_map.get(selected_camera_name)
        if selected_camera_index is None:
            messagebox.showerror("Error", "Invalid camera name selected.")
            return

        x = start_capture(self.controller.active_name, selected_camera_index)
        self.controller.num_of_images = x
        self.numimglabel.config(text=f"Number of images captured = {x}")
        if x >= 300:
            messagebox.showinfo("Capture Complete", "300 images have been captured!")

            
    def trainmodel(self):
        if self.controller.num_of_images < 300:
            messagebox.showerror("ERROR", "Not enough data, capture at least 300 images!")
            return

        data_directory = os.path.join(os.getcwd(), "data")

        # path fortrained model
        model_save_path = 'face_recognition_model.h5'
        train_classifier(data_directory, model_save_path)

        messagebox.showinfo("SUCCESS", "The model has been successfully trained!")
        self.controller.show_frame("PageFour")


threshold = 0.7  # lower values make recognition more strict, higher values make it more lenient
def load_images(base_dir, subfolder):
        images = []
        for i in range(1, 301):
            filename = f"{subfolder}_{i}.jpg"
            image_path = os.path.join(base_dir, subfolder, filename)
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                if img is not None:
                    images.append(img)
            else:
                break

        return images

class PageFour(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.recognition_thread = None
        self.face_detector = MTCNN(min_face_size=20)
        self.face_recognizer = InceptionResnetV1(pretrained='vggface2').eval()  

        label = tk.Label(self, text="Face Recognition", font='Helvetica 16 bold')
        label.grid(row=0, column=0, sticky="ew")

        recognition_button = tk.Button(self, text="Face Recognition", command=self.open_webcam,
                                       fg="#ffffff", bg="#263942", padx=20, pady=10, font=("Helvetica", 14))
        recognition_button.grid(row=1, column=0, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)

        log_button = tk.Button(self, text="View Attendance Log", command=lambda: controller.show_frame("AttendanceLogPage"),
                               fg="#ffffff", bg="#263942", padx=20, pady=10, font=("Helvetica", 14))
        log_button.grid(row=2, column=0, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)

    def log_attendance(self, student_names):
        csv_file = "attendance_log.csv"
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Write attendance data to spreadsheet
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            for student_name in student_names:
                writer.writerow([current_datetime, student_name])

        messagebox.showinfo("Attendance Logged", f"Attendance for {student_name} logged successfully.")

    def open_webcam(self):
        if self.recognition_thread is None or not self.recognition_thread.is_alive():
            self.recognition_thread = threading.Thread(target=self.run_face_recognition)
            self.recognition_thread.daemon = True  # Daemonize the thread
            self.recognition_thread.start()
    
    def run_face_recognition(self):
        try:
            if not hasattr(self, 'face_recognizer'):
                self.face_recognizer = load_face_recognition_model()

            known_faces = []
            known_names = []
            for name in names:
                known_faces += load_images('./data', name)  # Use the load_images function
                known_names.extend([name] * len(known_faces))

            cap = cv2.VideoCapture(0)

            while True:
                ret, frame = cap.read()

                if not ret:
                    continue  

                if frame is None:
                    continue  

                boxes, _ = self.face_detector.detect(frame)

                if boxes is None:
                    continue

                for box in boxes:
                    x, y, w, h = map(int, box)

                    detected_face = frame[y:y+h, x:x+w]
                    detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB)
                    detected_face = cv2.resize(detected_face, (160, 160))

                    x = max(0, x - int(0.2 * w)) 
                    y = max(0, y - int(0.2 * h))
                    w = min(frame.shape[1] - x, int(1.4 * w)) 
                    h = min(frame.shape[0] - y, int(1.4 * h))

                    predicted_name = self.recognize_face(known_faces, known_names, detected_face)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, predicted_name, (x, y - 10), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.imshow("Face Recognition", frame)

                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        except cv2.error as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


    def recognize_face(self, known_faces, known_names, detected_face):
        # Convert the detected face to a PyTorch tensor
        detected_face = (detected_face / 255.0 - 0.5) * 2.0  # Normalize to [-1, 1]
        detected_face = torch.from_numpy(detected_face.transpose((2, 0, 1))).unsqueeze(0).float()
        detected_embedding = self.face_recognizer(detected_face)

        closest_face = None
        min_distance = float('inf')

        # Compare the detected face with known faces and find the closest match
        for i, known_face in enumerate(known_faces):
            known_embedding = self.face_recognizer(torch.from_numpy(known_face).unsqueeze(0).float())
            distance = torch.norm(detected_embedding - known_embedding)

            if distance < min_distance:
                min_distance = distance
                closest_face = known_names[i]

        # Return the recognized name (or "Unknown" if no match is found)
        return closest_face if min_distance < threshold else "Unknown"


class AttendanceLogPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        log_label = tk.Label(self, text="Attendance Log", font=controller.title_font, fg="#263942")
        log_label.pack(pady=10)
        log_text = tk.Text(self, wrap=tk.WORD, width=50, height=10)
        log_text.pack(padx=20, pady=10)

        with open("attendance_log.csv", mode='r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                log_text.insert(tk.END, f"{row[0]} - {row[1]}\n")

        home_button = tk.Button(self, text="Back to Home", fg="#ffffff", bg="#263942",
                                command=lambda: controller.show_frame("StartPage"))
        home_button.pack(pady=10)


class MultiFaceDetectionPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.detected_faces = []

        label = tk.Label(self, text="Multiple Face Detection", font='Helvetica 16 bold')
        label.pack()

        detect_button = tk.Button(self, text="Start Face Detection", command=self.start_face_detection,
                                  fg="#ffffff", bg="#263942", padx=20, pady=10, font=("Helvetica", 14))
        detect_button.pack()

        home_button = tk.Button(self, text="Back to Home", fg="#ffffff", bg="#263942",
                                command=lambda: controller.show_frame("StartPage"), padx=20, pady=10, font=("Helvetica", 14))
        home_button.pack()

    def log_attendance(self):
        if not self.detected_faces:
            messagebox.showinfo("Attendance", "No faces detected.")
            return

        csv_file = "attendance_log.csv"
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            for student_name in self.detected_faces:
                writer.writerow([current_datetime, student_name])

        messagebox.showinfo("Attendance Logged", f"Attendance for {', '.join(self.detected_faces)} logged successfully.")
        self.detected_faces = []

    def start_face_detection(self):
        self.detected_faces = main_app(self.controller.active_name)
        self.log_attendance()


if __name__ == "__main__":
    app = MainUI()
    app.iconphoto(False, tk.PhotoImage(file='icon.ico'))
    app.mainloop()