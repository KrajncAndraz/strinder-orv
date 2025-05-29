import base64
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from datetime import datetime
import numpy as np
import cv2
import subprocess

# Dummy database of users' "face data"
user_face_db = {}

def save_face_setup(user_id, images):
    SAVE_DIR = os.path.join('faces', str(user_id))
    os.makedirs(SAVE_DIR, exist_ok=True)
    try:
        # Preprocesiraj slike (crop + augmentacija)
        procesirane_slike = preprocesiranje_slik(images)
        for idx, img in enumerate(procesirane_slike):
            filename = f"{idx:05d}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            # Pretvori RGB nazaj v BGR za shranjevanje z OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, img_bgr)
        train_user(user_id)  # Začni treniranje modela
        return True
    except Exception as e:
        print("Error during setup:", str(e))
        return False


def verify_face_image(user_id, image):
    try:
        save_dir = os.path.join('verification_images', str(user_id))
        os.makedirs(save_dir, exist_ok=True)
        img_path = os.path.join(save_dir, 'verify.jpg')
        img_bytes = base64.b64decode(image)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, success = obrezi(img)
        if not success:
            print("Nisem zaznal obraza.")
            return False    
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, img_bgr)
        status = subprocess.run(['python', 'verify_user.py', str(user_id), img_path])
        if status.returncode == 0:
            return True
        else:
            return False
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
                print("Nisem zaznal obraza.")
                continue    
            img, original_image = augumentiraj(img)
            procesirane_slike.append(img)
            procesirane_slike.append(original_image)
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
        original_image = image
        augumented_image = image

        #1.augmentacija - rotacija 50% sansa
        izvedi_rotacijo = np.random.uniform(0,10)
        if izvedi_rotacijo <= 5:
            angle = np.random.uniform(-10, 10)
            augumented_image = rotiraj_sliko(augumented_image, angle)

        #2.augmentacija - vertikalni flip 100% sansa
        augumented_image = vertikalni_flip(augumented_image)

        #3.augmentacija - sprememba svetlosti 30% sansa
        izvedi_rotacijo = np.random.uniform(0,10)
        if izvedi_rotacijo <= 3:
            delta = np.random.randint(-30, 30)
            augumented_image = spremeni_svetlost(augumented_image, delta)

        #4.augmentacija - bluranje 70% sansa
        izvedi_rotacijo = np.random.uniform(0,10)
        if izvedi_rotacijo <= 7:
            velikost_jedra = 3
            augumented_image = bluraj(augumented_image, velikost_jedra)

        return augumented_image, original_image
    except Exception as e:
        print("Napaka pri augmentaciji slike:", str(e))
        return image
    

def rotiraj_sliko(image, angle_degrees):


    angle_rad = np.deg2rad(angle_degrees)
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2

    rotated = np.zeros_like(image)

    for y in range(h):
        for x in range(w):
            x_shifted = x - center_x
            y_shifted = y - center_y

            src_x = int(np.cos(angle_rad) * x_shifted + np.sin(angle_rad) * y_shifted + center_x)
            src_y = int(-np.sin(angle_rad) * x_shifted + np.cos(angle_rad) * y_shifted + center_y)

            if 0 <= src_x < w and 0 <= src_y < h:
                rotated[y, x] = image[src_y, src_x]

    return rotated

def vertikalni_flip(image):
    h, w = image.shape[:2]
    flipped = np.zeros_like(image)
    for y in range(h):
        for x in range(w):
            flipped[y, x] = image[y, w - 1 - x]
    return flipped

def spremeni_svetlost(image, delta=30):
    #delta > 0 - svetlo
    #delta < 0 - temno

    h, w = image.shape[:2]
    nova_slika = np.zeros_like(image)
    for y in range(h):
        for x in range(w):
            rgb = image[y, x].astype(np.int16)
            nova_barva = np.clip(rgb + delta, 0, 255)
            nova_slika[y, x] = nova_barva.astype(np.uint8)
    return nova_slika

def bluraj(image, velikost_jedra=3):
    h, w, c = image.shape
    padding = velikost_jedra // 2
    # Razširi robove slike, da lahko obdeluješ tudi robne piksle
    padded = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
    blurana_slika = np.zeros_like(image)
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                okolica = padded[y:y+velikost_jedra, x:x+velikost_jedra, ch]
                blurana_slika[y, x, ch] = np.mean(okolica)
    return blurana_slika.astype(np.uint8)

def train_user(user_id):
    subprocess.run(['python', 'train_user_model.py', str(user_id)])

def verify_user(user_id, image_path):
    subprocess.run(['python', 'verify_user.py', str(user_id), str(image_path)])
