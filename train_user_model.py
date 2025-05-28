# requirements: pip install keras-facenet scikit-learn opencv-python numpy

import os
import sys
import glob
import shutil
import numpy as np
import cv2
from keras_facenet import FaceNet
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib

def load_images_from_folder(folder):
    images = []
    for filename in glob.glob(os.path.join(folder, '*.jpg')):
        img = cv2.imread(filename)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return images

def get_embeddings(images, embedder):
    embeddings = []
    for img in images:
        # FaceNet expects 160x160 RGB images
        img_resized = cv2.resize(img, (160, 160))
        emb = embedder.embeddings([img_resized])[0]
        embeddings.append(emb)
    return np.array(embeddings)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python train_user_model.py <user_id>")
        sys.exit(1)
    USER_ID = sys.argv[1]
    FACES_DIR = f'faces/{USER_ID}'
    NEGATIVE_DIR = 'train_faces/0'
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

        # Labels: 1 for user, 0 for others
        X_images = pos_images + neg_images
        y = np.array([1]*len(pos_images) + [0]*len(neg_images))

        # Check that there are at least two classes
        if len(set(y)) < 2:
            print("Error: Need at least two classes (positive and negative images) for training.")
            sys.exit(1)

        # Load FaceNet
        embedder = FaceNet()
        X = get_embeddings(X_images, embedder)

        # SVM with grid search for hyperparameter tuning
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        clf = GridSearchCV(SVC(probability=True), param_grid, cv=3)
        clf.fit(X, y)

        print("Best parameters:", clf.best_params_)
        print(classification_report(y, clf.predict(X)))

        # Save model
        model_path = os.path.join(MODEL_DIR, f'{USER_ID}_model.pkl')
        joblib.dump(clf.best_estimator_, model_path)
        print(f"Model saved to {model_path}")

        # Delete user images
        shutil.rmtree(FACES_DIR)
        print(f"Deleted user images in {FACES_DIR}")

    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)