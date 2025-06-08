import sys
import os
import cv2
import numpy as np
from tensorflow import keras
from face_logic import obrezi

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img , success = obrezi(img)  # Crop the face using the existing function
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    return img

def verify_user(user_id, image_path, threshold=0.5):
    model_path = os.path.join('models', f'{user_id}_model.keras')
    if not os.path.exists(model_path):
        print(f"Model for user {user_id} does not exist.")
        return False

    if not os.path.exists(image_path):
        print("Image does not exist.")
        return False

    model = keras.models.load_model(model_path)
    img = load_and_preprocess_image(image_path)
    if img is None:
        print("Error loading image.")
        return None
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prob = model.predict(img)[0][0]    # Get probability from sigmoid output
    print(f"Probability user is genuine: {prob:.2f}")
    return prob > threshold

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python verify_user.py <user_id> <image_path>")
        sys.exit(1)
    user_id = sys.argv[1]
    image_path = sys.argv[2]
    result = verify_user(user_id, image_path)
    if result is None:
        print("Verification failed.")
    else:
        print("Verification successful." if result else "Verification failed.")
    print("Result:", result)