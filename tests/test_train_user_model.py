import os
import shutil
import numpy as np
import cv2
import tempfile
import pytest
from train_user_model import load_images_from_folder, build_model, grid_search

def create_dummy_image(path, color=(255, 255, 255)):
    path = str(path)  # Ensure string path for OpenCV
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)
    img = np.full((10, 10, 3), color, dtype=np.uint8)
    cv2.imwrite(path, img)
    assert os.path.exists(path), f"Image file {path} was not created"

def test_load_images_from_folder(tmp_path):
    img_path = tmp_path / "img1.jpg"
    create_dummy_image(str(img_path))
    images = load_images_from_folder(str(tmp_path))
    assert len(images) == 1
    assert images[0].shape == (10, 10, 3)
    assert np.allclose(images[0].max(), 1.0)  # Normalized

def test_build_model():
    model = build_model((10, 10, 3))
    assert model is not None
    assert hasattr(model, "fit")
    assert model.input_shape[1:] == (10, 10, 3)

def test_grid_search_returns_model_and_params():
    # Create dummy data
    X = np.random.rand(20, 10, 10, 3).astype(np.float32)
    y = np.array([1]*10 + [0]*10)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model, params = grid_search(X_train, y_train, X_val, y_val, (10, 10, 3), "testuser")
    assert model is not None
    assert isinstance(params, dict)
    assert "num_filters" in params and "learning_rate" in params

def test_end_to_end_training(tmp_path):
    # Setup directories
    faces_dir = tmp_path / "faces" / "user1"
    neg_dir = tmp_path / "train_faces"
    model_dir = tmp_path / "models"
    os.makedirs(faces_dir)
    os.makedirs(neg_dir)
    os.makedirs(model_dir)

    # Create dummy images
    for i in range(2):
        create_dummy_image(str(faces_dir / f"pos_{i}.jpg"), color=(255, 0, 0))
        create_dummy_image(str(neg_dir / f"neg_{i}.jpg"), color=(0, 255, 0))

    # Patch paths in the script
    import train_user_model as tum
    tum.MODEL_DIR = str(model_dir)
    tum.FACES_DIR = str(faces_dir)
    tum.NEGATIVE_DIR = str(neg_dir)

    # Run main logic as a function (simulate)
    pos_images = load_images_from_folder(str(faces_dir))
    neg_images = load_images_from_folder(str(neg_dir))
    assert len(pos_images) == 2
    assert len(neg_images) == 2

    pos_labels = [1] * len(pos_images)
    neg_labels = [0] * len(neg_images)
    X = np.array(pos_images + neg_images)
    y = np.array(pos_labels + neg_labels)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    input_shape = X_train.shape[1:]
    model, best_params = grid_search(X_train, y_train, X_val, y_val, input_shape, "user1")
    model_path = os.path.join(model_dir, "user1_model.keras")
    model.save(model_path)
    assert os.path.exists(model_path)