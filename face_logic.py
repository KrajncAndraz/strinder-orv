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
        for idx, img_base64 in enumerate(images):
            filename = f"{user_id}_setup_{idx+1}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            with open(filepath, "wb") as f:
                f.write(base64.b64decode(img_base64))
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
            img = obrezi(img)    
            img = augumentiraj(img)
            procesirane_slike.append(img)
        except Exception as e:
            print("Napaka pri obdelavi slike:", str(e))
    return procesirane_slike

def obrezi(img, crop_size=(160, 160)):
    h, w = img.shape[:2]
    ch, cw = crop_size
    start_y = max((h - ch) // 2, 0)
    start_x = max((w - cw) // 2, 0)
    end_y = start_y + ch
    end_x = start_x + cw
    return img[start_y:end_y, start_x:end_x]

def augumentiraj(image):
    try:
        #zaenkrat sam gaussian blur !!!!ne dotikaj se to je delo za clana 1
        augmented_img = cv2.GaussianBlur(image, (5, 5), 0)
        return augmented_img
    except Exception as e:
        print("Napaka pri augmentaciji slike:", str(e))
        return image