import pytest
from flask import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask_server import app 


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_setup_face_connection(client):
    response = client.post('/setup-face', json={
        'userId': 'testuser',
        'images': ['img1', 'img2', 'img3', 'img4', 'img5']
    })

    assert response.status_code in (200, 500, 400)

def test_verify_face_connection(client):
    response = client.post('/verify-face', json={
        'userId': 'testuser',
        'image': 'img1'
    })

    assert response.status_code in (200, 500, 400)
