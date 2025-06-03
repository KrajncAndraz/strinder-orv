import os
import sys
import glob
import numpy as np
import cv2
import shutil

def load_images_from_folder(folder):
    images = []
    for filename in glob.glob(os.path.join(folder, '*.jpg')):
        img = cv2.imread(filename)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return images

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python train_user_model.py <user_id>")
        sys.exit(1)
    USER_ID = sys.argv[1]
    FACES_DIR = f'faces/{USER_ID}'
    NEGATIVE_DIR = 'train_faces/'
    MODEL_DIR = 'models'
    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        # Load images
        pos_images = load_images_from_folder(FACES_DIR)
        neg_images = load_images_from_folder(NEGATIVE_DIR)

        if len(pos_images) == 0:
            print(f"Error: No positive images found in {FACES_DIR}")
            sys.exit(1)
        if len(neg_images) == 0:
            print(f"Error: No negative images found in {NEGATIVE_DIR}")
            sys.exit(1)

       

        # Delete user images
        shutil.rmtree(FACES_DIR)
        print(f"Deleted user images in {FACES_DIR}")

    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)