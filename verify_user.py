import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import joblib
from keras_facenet import FaceNet

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 160))
    return img

def verify_user(user_id, image_path, threshold=0.5):
    model_path = os.path.join('models', f'{user_id}_model.keras')
    if not os.path.exists(model_path):
        print(f"Model za uporabnika {user_id} ne obstaja.")
        return False
    
    if not os.path.exists(image_path):
        print(f"Slika ne obstaja.")
        return False

    clf = joblib.load(model_path)
    embedder = FaceNet()
    img = load_and_preprocess_image(image_path)
    if img is None:
        print("Napaka pri nalaganju slike.")
        return None
    emb = embedder.embeddings([img])[0].reshape(1, -1)
    prob = clf.predict_proba(emb)[0][1]  # verjetnost, da je user
    print(f"Verjetnost, da je pravi uporabnik: {prob:.2f}")
    return prob > threshold

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Uporaba: python verify_user.py <user_id> <pot_do_slike>")
        sys.exit(1)
    user_id = sys.argv[1]
    image_path = sys.argv[2]
    result = verify_user(user_id, image_path)
    if result is None:
        print("Verifikacija neuspešna.")
    else:
        print("Verifikacija uspešna.")
    print("Rezultat:", result)