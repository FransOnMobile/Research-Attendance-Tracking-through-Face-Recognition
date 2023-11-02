import os, cv2
import numpy as np 
import shutil, csv
import tkinter as tk
from tkinter import messagebox
from tkinter import font as tkfont
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from create_classifier import train_classifier
from create_dataset import start_capture
import dlib

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

        self.frames = {
            "StartPage": StartPage(parent=container, controller=self),
            "ImageCapturePage": ImageCapturePage(parent=self, controller=self),
            "ModelTrainingPage": ModelTrainingPage(parent=container, controller=self),
            "FaceRecognitionPage": FaceRecognitionPage(parent=container, controller=self),
        }

        for F in (StartPage, ImageCapturePage, ModelTrainingPage, PageTwo, PageFour, FaceRecognitionPage):
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

    def get_image_count_for_student(self, student_name):
        student_folder = f'./data/{student_name}'
        if os.path.exists(student_folder):
            images = [file for file in os.listdir(student_folder) if file.endswith('.jpg')]
            return len(images)
        return 0 
    
    def get_selected_student_name(self):
        return self.frames["ModelTrainingPage"].selected_student_name


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.student_list_var = tk.StringVar()
        self.selected_student_name = None

        title_label = tk.Label(self, text="Attendance Tracking System", font=("Helvetica", 24), fg="#263942")
        title_label.pack(pady=20)
        
        student_list_frame = tk.Frame(self, padx=20, pady=20)
        student_list_frame.pack()
        
        list_header = tk.Label(student_list_frame, text="Registered Students", font=("Helvetica", 16), fg="#263942")
        list_header.grid(row=0, column=0, columnspan=2, pady=10)

        # frame for buttons
        button_frame = tk.Frame(self, padx=20, pady=20)
        button_frame.pack()
        
        # Listbox to display the students
        self.student_listbox = tk.Listbox(student_list_frame, listvariable=self.student_list_var, selectmode=tk.SINGLE, width=40, height=10)
        self.student_listbox.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        # buttons
        check_user_button = tk.Button(button_frame, text="Check a Student", fg="white", bg="#263942",
                                      command=lambda: controller.show_frame("PageTwo"), padx=20, pady=10, font=("Helvetica", 14))
        clear_data_button = tk.Button(button_frame, text="Clear Data", fg="#ffffff", bg="#FF0000",
                                      command=self.confirm_clear_data, padx=20, pady=10, font=("Helvetica", 14))
        quit_button = tk.Button(button_frame, text="Quit", fg="#263942", bg="white", command=self.on_closing,
                                padx=20, pady=10, font=("Helvetica", 14))
        capture_image_button = tk.Button(self, text="Capture Images", command=lambda: controller.show_frame("ImageCapturePage"),
                                         fg="#ffffff", bg="#263942", padx=20, pady=10, font=("Helvetica", 14))
        train_model_button = tk.Button(self, text="Train Model", command=lambda: controller.show_frame("ModelTrainingPage"),
                                       fg="#ffffff", bg="#263942", padx=20, pady=10, font=("Helvetica", 14))
        recognize_face_button = tk.Button(self, text="Face Recognition",
                                          command=lambda: controller.show_frame("FaceRecognitionPage"),
                                          fg="#ffffff", bg="#263942", padx=20, pady=10, font=("Helvetica", 14))
        recognize_face_button.pack(pady=10)
        train_model_button.pack(pady=10)
        capture_image_button.pack(pady=10)  
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


    def on_closing(self):
        if messagebox.askokcancel("Quit", "Are you sure?"):
            global names
            with open("nameslist.txt", "w") as f:
                for i in names:
                    f.write(i + " ")
            self.controller.destroy()
          
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

        sorted_names = sorted(names)
        for name in sorted_names:
            self.student_listbox.insert(tk.END, name)

##########  start of image capture page #############

class ImageCapturePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.student_name_label = tk.Label(self, text="Enter Student Name:", font=("Helvetica", 12))
        self.student_name_label.pack()

        self.student_name_entry = tk.Entry(self)
        self.student_name_entry.pack()

        self.capture_button = tk.Button(self, text="Capture", command=self.capture_images)
        self.capture_button.pack()

        self.home_button = tk.Button(self, text="Home", command=lambda: controller.show_frame("StartPage"))
        self.home_button.pack()

    def capture_images(self):
        student_name = self.student_name_entry.get()
        if not student_name:
            messagebox.showerror("Error", "Please enter a student name.")
            return

        selected_camera_index = 0  # Set the camera index according to your setup
        num_images = 300

        num_of_images = start_capture(student_name, selected_camera_index)

        if num_of_images is not None:
            if num_of_images >= num_images:
                messagebox.showinfo("Info", f"{num_images} images of {student_name} captured.")
            else:
                messagebox.showinfo("Info", f"Captured {num_of_images} images for student '{student_name}'.")

class ModelTrainingPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.selected_student_name = None

        train_model_button = tk.Button(self, text="Train Model", command=self.train_model,
                                       fg="#ffffff", bg="#263942", padx=20, pady=10, font=("Helvetica", 14))
        train_model_button.pack(pady=10)

        self.home_button = tk.Button(self, text="Home", command=lambda: controller.show_frame("StartPage"))
        self.home_button.pack()

        students = [student for student in os.listdir('./data') if os.path.isdir(os.path.join('./data', student))]

        self.checkboxes = []
        for student_name in students:
            var = tk.IntVar()
            checkbox = tk.Checkbutton(self, text=student_name, variable=var,
                                      command=lambda name=student_name: self.select_student(name))
            checkbox.pack()
            self.checkboxes.append((checkbox, student_name))

    def check_if_trained(self, student_name):
        flag_file_path = f'./data/{student_name}/{student_name}_trained.txt'
        return os.path.exists(flag_file_path)

    def train_model(self):
        student_name = self.controller.get_selected_student_name()

        if not student_name:
            messagebox.showerror("Error", "Please select a student.")
            return

        image_count = self.controller.get_image_count_for_student(student_name)

        if image_count < 150:
            messagebox.showerror("Error", "Not enough data, capture at least 150 images!")
            return

        if not self.check_if_trained(student_name):
            print(student_name)
            data_directory = os.path.join(os.getcwd(), "data")

            try:
                train_classifier(data_directory)

                # After successful training, create the flag file to indicate the training is complete
                flag_file_path = f'./data/{student_name}/{student_name}_trained.txt'
                with open(flag_file_path, 'w') as flag_file:
                    flag_file.write("Trained")
                
                messagebox.showinfo("Success", f"The model for {student_name} has been successfully trained!")
                self.controller.show_frame("ModelTrainingPage")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during model training: {str(e)}")
        else:
            messagebox.showwarning("Already Trained", f"{student_name} has already been trained.")

    def select_student(self, student_name):
        if student_name == self.selected_student_name:
            self.selected_student_name = None
        else:
            self.selected_student_name = student_name


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

class PageFour(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.recognition_thread = None
        self.face_detector = MTCNN(min_face_size=20)
        self.face_recognizer = InceptionResnetV1(pretrained='vggface2').eval()  

        label = tk.Label(self, text="Face Recognition", font='Helvetica 16 bold')
        label.grid(row=0, column=0, sticky="ew")


class FaceRecognitionPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.detector = dlib.get_frontal_face_detector()
        self.face_classifier = InceptionResnetV1(pretrained='vggface2').eval()
        self.students_embeddings = self.load_students_embeddings()

        recognize_button = tk.Button(self, text="Recognize", command=self.recognize_face,
                                     fg="#ffffff", bg="#263942", padx=20, pady=10, font=("Helvetica", 14))
        recognize_button.pack(pady=10)

        self.home_button = tk.Button(self, text="Home", command=lambda: controller.show_frame("StartPage"))
        self.home_button.pack()

    def load_students_embeddings(self):
        # Load pre-trained student face embeddings
        students = [student for student in os.listdir('./data') if os.path.isdir(os.path.join('./data', student))]
        students_embeddings = {}

        for student_name in students:
            # Load embeddings for each student
            embeddings_file = f'./data/{student_name}/{student_name}_embeddings.npy'
            if os.path.exists(embeddings_file):
                students_embeddings[student_name] = np.load(embeddings_file)

        return students_embeddings
    
    def recognize_face(self):
        camera = cv2.VideoCapture(0)
        while True:
            ret, frame = camera.read()

            # Convert the frame to RGB for dlib and facenet
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces using dlib
            faces = self.detector(frame_rgb)

            if len(faces) > 0:
                for face in faces:
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Crop face and convert to RGB
                    face_crop = frame_rgb[y:y+h, x:x+w]
                    face_pil = Image.fromarray(face_crop)

                    # Resize the face to fit the facenet model
                    face_resized = face_pil.resize((160, 160), Image.ANTIALIAS)

                    # Generate embeddings using facenet
                    face_tensor = self.normalize(np.array(face_resized))
                    embeddings = self.face_classifier(face_tensor.unsqueeze(0))

                    # Compare embeddings against student embeddings
                    recognized = False
                    for student, student_embeddings in self.students_embeddings.items():
                        distance = np.linalg.norm(embeddings.detach().numpy() - student_embeddings, axis=1)
                        if np.min(distance) < 0.7:  # Set an appropriate threshold
                            recognized = True
                            cv2.putText(frame, student, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                            break

                    if not recognized:
                        cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()

    def load_students_embeddings(self):
        students_embeddings = {}

        # Load embeddings for each student
        for student in os.listdir('./data'):
            student_path = os.path.join('./data', student)
            if os.path.isdir(student_path):
                embeddings_file = os.path.join(student_path, f'{student}_embeddings.npy')
                if os.path.exists(embeddings_file):
                    students_embeddings[student] = np.load(embeddings_file)

        return students_embeddings

    def normalize(self, img):
        return (img / 255.0 - 0.5) / 0.5


if __name__ == "__main__":
    app = MainUI()
    app.iconphoto(False, tk.PhotoImage(file='icon.ico'))
    app.mainloop()