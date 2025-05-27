import base64
import os
from datetime import datetime
import numpy as np
import cv2

# Dummy database of users' "face data"
user_face_db = {}

def save_face_setup(user_id, images):
    SAVE_DIR = 'faces'
    os.makedirs(SAVE_DIR, exist_ok=True)
    try:
        # Preprocesiraj slike (crop + augmentacija)
        procesirane_slike = preprocesiranje_slik(images)
        for idx, img in enumerate(procesirane_slike):
            filename = f"{user_id}_setup_{idx+1}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            # Pretvori RGB nazaj v BGR za shranjevanje z OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, img_bgr)
        return True
    except Exception as e:
        print("Error during setup:", str(e))
        return False


def verify_face_image(user_id, image):
    try:
        save_dir = os.path.join('verification_images', str(user_id))
        os.makedirs(save_dir, exist_ok=True)
        img_path = os.path.join(save_dir, 'verify.jpg')
        with open(img_path, "wb") as fh:
            fh.write(base64.b64decode(image))
        return True
    except Exception as e:
        print("Error during verification:", str(e))
        return False


def preprocesiranje_slik(images_base64):
    procesirane_slike = []
    for img_base64 in images_base64:
        try:
            img_bytes = base64.b64decode(img_base64)
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                print("Napaka pri dekodiranju slike.")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img, success = obrezi(img)
            if not success:
                print("Napaka pri obrezi slike.")
                continue    
            img = augumentiraj(img)
            procesirane_slike.append(img)
        except Exception as e:
            print("Napaka pri obdelavi slike:", str(e))
    return procesirane_slike

def obrezi(img, crop_size=(160, 200)):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Prvi detektiran obraz
        face_img = img[y:y+h, x:x+w]
        # Resize na željeno velikost
        face_img = cv2.resize(face_img, crop_size)
        return face_img, True
    else:
        return img, False

def augumentiraj(image):
    try:
        #zaenkrat sam gaussian blur !!!!ne dotikaj se to je delo za clana 1
        augmented_img = cv2.GaussianBlur(image, (9, 9), 0)
        return augmented_img
    except Exception as e:
        print("Napaka pri augmentaciji slike:", str(e))
        return image