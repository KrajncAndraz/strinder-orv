import base64
import os
from datetime import datetime

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
