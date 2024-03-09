import os
import tkinter as tk
from tkinter import ttk, messagebox
from subprocess import Popen
import customtkinter as ctk
from face_registration import register_face
import pandas as pd
from datetime import datetime
import json

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

        modify_periods_button = ctk.CTkButton(self, text="Modify Periods", command=master.open_modify_periods)
        modify_periods_button.pack(padx=10, pady=10, fill='x')

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

        # Initialize class periods as an empty dictionary
        self.class_periods = {}

    def open_detect(self):
        try:
            Popen(["python", "detect.py"])
        except Exception as e:
            messagebox.showerror("Error", f"Error running detect.py: {str(e)}")

    def open_train(self):
        import train_v2
        try:
            train_v2.train_faces()  # Call the train_faces function from train_v2.py
            messagebox.showinfo("Training", "Training completed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error running train_faces: {str(e)}")

    def view_attendance(self):
        try:
            logs_dir = "Attendance Logs"
            date_today = datetime.now().date().strftime('%Y-%m-%d')
            file_path = os.path.join(logs_dir, f"Attendance_{date_today}.xlsx")

            if os.path.exists(file_path):
                df = pd.read_excel(file_path)
                messagebox.showinfo("View Attendance", f"Attendance data for {date_today}:\n\n{df}")
            else:
                messagebox.showwarning("View Attendance", f"No attendance data available for {date_today}.")
        except Exception as e:
            messagebox.showerror("Error", f"Error viewing attendance: {str(e)}")

    def open_register(self):
        register_page = RegisterPage(self)
        register_page.pack(fill='both', expand=True)

    def open_modify_periods(self):
        modify_page = ModifyPeriodsPage(self, self.class_periods)
        modify_page.pack(fill='both', expand = True)

class ModifyPeriodsPage(tk.Frame):
    def __init__(self, master, class_periods):
        super().__init__(master)

        self.master = master  # Save a reference to the master
        self.class_periods = class_periods  # Store class periods dictionary

        modify_label = ctk.CTkLabel(self, text="Modify Class Periods", font=("Helvetica", 16, "bold"), text_color="black")
        modify_label.pack(pady=20)

        self.days_combobox = ttk.Combobox(self, values=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        self.days_combobox.set("Monday")
        self.days_combobox.pack(pady=10)

        self.periods_frame = tk.Frame(self)
        self.periods_frame.pack(pady=10)

        self.load_periods()

        self.start_entry = ctk.CTkEntry(self)
        self.start_entry.insert(0, "Start Time (HH:MM)")
        self.start_entry.pack(pady=5)
        
        self.end_entry = ctk.CTkEntry(self)
        self.end_entry.insert(0, "End Time (HH:MM)")
        self.end_entry.pack(pady=5)
        
        self.name_entry = ctk.CTkEntry(self)
        self.name_entry.insert(0, "Period Name")
        self.name_entry.pack(pady=5)

        add_button = ctk.CTkButton(self, text="Add Class Period", command=self.add_period, text_color="black")
        add_button.pack(padx=10, pady=10, fill='x')

        delete_button = ctk.CTkButton(self, text="Delete Class Period", command=self.delete_period, text_color="black")
        delete_button.pack(padx=10, pady=10, fill='x')

        save_button = ctk.CTkButton(self, text="Save Changes", command=self.save_changes, text_color="black")
        save_button.pack(padx=10, pady=10, fill='x')

    def load_periods(self):
        # Clear existing widgets
        for widget in self.periods_frame.winfo_children():
            widget.destroy()

        # Load class periods for the selected day
        selected_day = self.days_combobox.get()
        if selected_day in self.class_periods:
            periods = self.class_periods[selected_day]
            for i, (start_time, end_time, period_name) in enumerate(periods):
                label_text = f"{start_time} - {end_time}: {period_name}"
                label = ctk.CTkLabel(self.periods_frame, text=label_text)
                label.grid(row=i, column=0, sticky="w")

    def add_period(self):
        # Add a new class period to the selected day
        selected_day = self.days_combobox.get()
        if selected_day not in self.class_periods:
            self.class_periods[selected_day] = []

        start_time = self.start_entry.get()
        end_time = self.end_entry.get()
        period_name = self.name_entry.get()

        # Validate input
        if not all([start_time, end_time, period_name]):
            messagebox.showwarning("Warning", "Please enter all details for the period.")
            return

        self.class_periods[selected_day].append((start_time, end_time, period_name))
        self.load_periods()

    def delete_period(self):
        # Delete the selected class period from the selected day
        selected_day = self.days_combobox.get()
        if selected_day in self.class_periods:
            periods = self.class_periods[selected_day]
            if periods:
                del periods[-1]  # Delete the last period
                self.load_periods()

    def save_changes(self):
        # Save the modified class periods to a JSON file
        try:
            with open('class_periods.json', 'w') as json_file:
                json.dump(self.class_periods, json_file, indent=4)
            messagebox.showinfo("Info", "Changes saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving changes: {str(e)}")

if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.mainloop()
