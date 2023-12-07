import tkinter as tk
from tkinter import ttk, messagebox
from subprocess import Popen
import customtkinter as ctk
from face_registration import register_face

# Set the appearance mode and color theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class Header(tk.Frame):
    def __init__(self, master):
        super().__init__(master)

        title_label = ctk.CTkLabel(self, text="PSHS-CMC Classroom Attendance Tracker", font=("Helvetica", 16, "bold"), text_color="black")
        title_label.pack(pady=10)

        subtitle_label = ctk.CTkLabel(self, text="A Research Project", font=("Helvetica", 12), text_color="black")
        subtitle_label.pack(pady=5)

class SideBar(tk.Frame):
    def __init__(self, master):
        super().__init__(master)

        detect_button = ctk.CTkButton(self, text="Run Face Detection", command=master.open_detect)
        detect_button.pack(padx=10, pady=10, fill='x')

        train_button = ctk.CTkButton(self, text="Run Face Training", command=master.open_train)
        train_button.pack(padx=10, pady=10, fill='x')

        attendance_button = ctk.CTkButton(self, text="View Attendance", command=master.view_attendance)
        attendance_button.pack(padx=10, pady=10, fill='x')

        register_button = ctk.CTkButton(self, text="Register Face", command=master.open_register)
        register_button.pack(padx=10, pady=10, fill='x')

class Footer(tk.Frame):
    def __init__(self, master):
        super().__init__(master)

        exit_button = ctk.CTkButton(self, text="Exit", command=master.destroy)
        exit_button.pack(pady=10, padx=20, fill='x')

class RegisterPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master)

        self.master = master  # Save a reference to the master

        register_label = ctk.CTkLabel(self, text="Register Your Face", font=("Helvetica", 16, "bold"), text_color="black")
        register_label.pack(pady=20)

        # Entry for student's name with a placeholder
        self.name_entry = ctk.CTkEntry(self)
        self.name_entry.insert(0, "Enter your name")
        self.name_entry.bind("<FocusIn>", self.clear_placeholder)
        self.name_entry.bind("<FocusOut>", self.restore_placeholder)
        self.name_entry.pack(pady=10)

        # Button to register face
        register_button = ctk.CTkButton(self, text="Register Face", command=self.register_face, text_color="black")
        register_button.pack(padx=10, pady=10, fill='x')

    def clear_placeholder(self, event):
        if self.name_entry.get() == "Enter your name":
            self.name_entry.delete(0, tk.END)
            self.name_entry.configure(text_color="white")  # Change text color when focused

    def restore_placeholder(self, event):
        if not self.name_entry.get():
            self.name_entry.insert(0, "Enter your name")
            self.name_entry.configure(text_color="grey")  # Change text color when not focused

    def register_face(self):
        student_name = self.name_entry.get()
        if student_name == "Enter your name":
            messagebox.showwarning("Warning", "Please enter your name.")
            return

        try:
            register_face(student_name)
            messagebox.showinfo("Success", "Face registered successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error registering face: {str(e)}")


class FaceRecognitionApp(tk.Tk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.title("Classroom Attendance Tracker")
        self.geometry("800x600")  # Set default size

        self.header = Header(self)
        self.side_bar = SideBar(self)
        self.footer = Footer(self)

        self.header.pack(fill='x')
        self.side_bar.pack(side='left', fill='y')
        self.footer.pack(side='bottom', fill='x')

    def open_detect(self):
        try:
            Popen(["python", "detect.py"])
        except Exception as e:
            messagebox.showerror("Error", f"Error running detect.py: {str(e)}")

    def open_train(self):
        try:
            Popen(["python", "train_v2.py"])
        except Exception as e:
            messagebox.showerror("Error", f"Error running train_v2.py: {str(e)}")

    def view_attendance(self):
        try:
            with open('attendance.csv', 'r', newline='') as csvfile:
                print(csvfile.read())
        except FileNotFoundError:
            messagebox.showwarning("Warning", "Attendance data file not found.")

        messagebox.showinfo("View Attendance", "Attendance data printed to console!")

    def open_register(self):
        register_page = RegisterPage(self)
        register_page.pack(fill='both', expand=True)

if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.mainloop()
