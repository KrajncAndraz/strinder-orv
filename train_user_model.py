import os
import sys
import glob
import numpy as np
import cv2
import shutil
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from face_logic import obrezi

def load_images_from_folder(folder):
    images = []
    for filename in glob.glob(os.path.join(folder, '*.jpg')):
        img = cv2.imread(filename)
        if img is not None:
            #img, success = obrezi(img)
            #if not success:
            #    continue  # Skip if face not detected
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype('float32') / 255.0  # Normalize to [0, 1]
            images.append(img)
    return images

def build_model(input_shape, num_filters=32, learning_rate=0.001):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        keras.layers.Conv2D(num_filters, (3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(num_filters*2, (3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def grid_search(X_train, y_train, X_val, y_val, input_shape, USER_ID):
    # Simple grid search over two hyperparameters
    best_acc = 0
    best_params = {}
    results = []
    for num_filters in [16, 32]:
        for lr in [0.001, 0.0005]:
            model = build_model(input_shape, num_filters=num_filters, learning_rate=lr)
            history = model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0, validation_data=(X_val, y_val))
            val_acc = history.history['val_accuracy'][-1]
            results.append((num_filters, lr, val_acc))
            if val_acc > best_acc:
                best_acc = val_acc
                best_params = {'num_filters': num_filters, 'learning_rate': lr}
                best_model = model
    # Log results for report
    os.makedirs('models/logs', exist_ok=True)
    with open(f'models/logs/hyperparam_log_for_{USER_ID}.txt', 'a') as f:
        for nf, lr, acc in results:
            f.write(f'num_filters={nf}, learning_rate={lr}, val_acc={acc:.4f}\n')
    return best_model, best_params


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
            
        # Prepare data
        pos_labels = [1] * len(pos_images)
        neg_labels = [0] * len(neg_images)
        X = np.array(pos_images + neg_images)
        y = np.array(pos_labels + neg_labels)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        # Model training with hyperparameter search
        input_shape = X_train.shape[1:]
        model, best_params = grid_search(X_train, y_train, X_val, y_val, input_shape, USER_ID)
        print(f"Best hyperparameters: {best_params}")

        # Save model
        model_path = os.path.join(MODEL_DIR, f'{USER_ID}_model.keras')
        model.save(model_path)
        print(f"Model saved to {model_path}")

        # Delete user images
        #shutil.rmtree(FACES_DIR)
        #print(f"Deleted user images in {FACES_DIR}")

    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)