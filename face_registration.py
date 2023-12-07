import cv2
import dlib
import os

def register_face(student_name):
    # Create a folder for the student's face images
    student_folder = f"faces/{student_name}"
    os.makedirs(student_folder, exist_ok=True)

    # Initialize the face detector and video capture
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(0)

    try:
        print("Please look directly at the camera.")
        for i in range(10):
            ret, frame = cap.read()
            if not ret:
                print("Error capturing image. Exiting.")
                break

            # Convert the frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = detector(gray_frame)

            # Save face images
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                face_image = frame[y:y+h, x:x+w]

                image_path = os.path.join(student_folder, f"{student_name}_{i + 1}.jpg")
                cv2.imwrite(image_path, face_image)

                print(f"Image {i + 1} captured.")

            cv2.imshow("Capture", frame)
            cv2.waitKey(500)

    finally:
        # Release the camera
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    student_name = input("Enter the student's name: ")
    register_face(student_name)
