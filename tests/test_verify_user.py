import os
import numpy as np
import cv2
import tempfile
import pytest
from unittest import mock
import verify_user

def create_dummy_image(path, color=(255, 255, 255)):
    path = str(path)  # Ensure string path for OpenCV
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)
    img = np.full((10, 10, 3), color, dtype=np.uint8)
    cv2.imwrite(path, img)
    assert os.path.exists(path), f"Image file {path} was not created"

def test_load_and_preprocess_image_valid(tmp_path):
    img_path = tmp_path / "img.jpg"
    create_dummy_image(str(img_path))
    # Patch obrezi to just return the image and True
    with mock.patch("face_logic.obrezi", return_value=(cv2.imread(str(img_path)), True)):
        img = verify_user.load_and_preprocess_image(str(img_path))
    assert img is not None
    assert img.shape == (10, 10, 3)
    assert np.allclose(img.max(), 1.0)

def test_load_and_preprocess_image_invalid(tmp_path):
    img_path = tmp_path / "nonexistent.jpg"
    img = verify_user.load_and_preprocess_image(str(img_path))
    assert img is None

def test_verify_user_model_missing(tmp_path):
    img_path = tmp_path / "img.jpg"
    create_dummy_image(str(img_path))
    result = verify_user.verify_user("nonexistent_user", str(img_path))
    assert result is False

def test_verify_user_image_missing(tmp_path):
    model_dir = tmp_path / "models"
    os.makedirs(model_dir)
    model_path = model_dir / "user1_model.keras"
    # Create an empty file to simulate model
    model_path.write_text("")
    with mock.patch("verify_user.keras.models.load_model"):
        result = verify_user.verify_user("user1", str(tmp_path / "noimg.jpg"))
    assert result is False

def test_verify_user_invalid_image(tmp_path):
    model_dir = tmp_path / "models"
    os.makedirs(model_dir)
    model_path = model_dir / "user1_model.keras"
    model_path.write_text("")
    with mock.patch("verify_user.keras.models.load_model"):
        # Patch load_and_preprocess_image to return None
        with mock.patch("verify_user.load_and_preprocess_image", return_value=None):
            result = verify_user.verify_user("user1", str(tmp_path / "img.jpg"))
    assert result is None

def test_verify_user_prediction(tmp_path):
    # Setup model and image
    model_dir = tmp_path / "models"
    os.makedirs(model_dir)
    model_path = model_dir / "user1_model.keras"
    model_path.write_text("")
    img_path = tmp_path / "img.jpg"
    create_dummy_image(str(img_path))
    # Patch model and prediction
    mock_model = mock.Mock()
    mock_model.predict.return_value = np.array([[0.8]])
    with mock.patch("verify_user.keras.models.load_model", return_value=mock_model):
        with mock.patch("verify_user.load_and_preprocess_image", return_value=np.ones((10, 10, 3), dtype=np.float32)):
            result = verify_user.verify_user("user1", str(img_path), threshold=0.5)
    assert result is True
    # Test below threshold
    mock_model.predict.return_value = np.array([[0.2]])
    with mock.patch("verify_user.keras.models.load_model", return_value=mock_model):
        with mock.patch("verify_user.load_and_preprocess_image", return_value=np.ones((10, 10, 3), dtype=np.float32)):
            result = verify_user.verify_user("user1", str(img_path), threshold=0.5)
    assert result is False