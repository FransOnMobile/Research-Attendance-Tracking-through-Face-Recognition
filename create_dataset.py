import os
import cv2

# Capture and save images for dataset
def start_capture(name, selected_camera_index, num_images=300):
    path = os.path.join("data", name)
    try:
        os.makedirs(path)
    except FileExistsError:
        print('Directory Already Created')

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    vid = cv2.VideoCapture(selected_camera_index)
    num_of_images = 0

    while num_of_images < num_images:
        ret, img = vid.read()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

        for x, y, w, h in faces:
            # Crop and save the dete    cted face
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
