import dlib
import cv2
import os

def start_capture(name, selected_camera_index, num_images=150):
    path = os.path.join("data", name)

    if not os.path.exists(path):
        try:
            os.makedirs(path)

            # Load the face detector from dlib
            detector = dlib.get_frontal_face_detector()

            vid = cv2.VideoCapture(selected_camera_index)
            num_of_images = 0

            while num_of_images < num_images:
                ret, img = vid.read()

                # Convert the image to grayscale for dlib face detection
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Detect faces using dlib
                faces = detector(gray_img)

                for face in faces:
                    # Extract the bounding box coordinates of the detected face
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()

                    # Crop and save the detected face
                    face_img = gray_img[y:y + h, x:x + w]
                    img_filename = f"{name}_{num_of_images + 1}.jpg"
                    img_path = os.path.join(path, img_filename)
                    cv2.imwrite(img_path, face_img)

                    num_of_images += 1

                cv2.imshow("Capture", img)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27 or num_of_images >= num_images:
                    break

            vid.release()
            cv2.destroyAllWindows()

            return num_of_images
        except OSError as e:
            print(f"Error creating directory: {e}")
            return None
    else:
        print(f"Directory {path} already exists.")
        return None
