import pytest
import base64
import requests
import os

BASE_URL = "http://localhost:8000"

TEST_CASES = [
    {
        "name": "Same Person: ID to Selfie",
        "user_id": "test_user_001",
        "image_type": "id_to_selfie",
        "image1_path": "test_images/id_photos/person1_id.jpg",
        "image2_path": "test_images/selfies/person1_selfie.jpg",
        "expected_match": True
    },
    {
        "name": "Same Person: Selfie to Selfie",
        "user_id": "test_user_002",
        "image_type": "selfie_to_selfie",
        "image1_path": "test_images/selfies/person1_selfie.jpg",
        "image2_path": "test_images/selfies/person1_selfie2.jpg",
        "expected_match": True
    },
    {
        "name": "Different People",
        "user_id": "test_user_003",
        "image_type": "id_to_selfie",
        "image1_path": "test_images/id_photos/person1_id.jpg",
        "image2_path": "test_images/selfies/person2_selfie.jpg",
        "expected_match": False
    },
    {
        "name": "No Face",
        "user_id": "test_user_004",
        "image_type": "id_to_selfie",
        "image1_path": "test_images/edge_cases/no_face.jpg",
        "image2_path": "test_images/selfies/person1_selfie.jpg",
        "expected_match": False
    },
]

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@pytest.mark.parametrize("case", TEST_CASES)
def test_face_verification(case):
    if not os.path.exists(case['image1_path']) or not os.path.exists(case['image2_path']):
        pytest.skip(f"Image files not found: {case['image1_path']} or {case['image2_path']}")

    image1_b64 = image_to_base64(case['image1_path'])
    image2_b64 = image_to_base64(case['image2_path'])

    request_data = {
        "user_id": case['user_id'],
        "image_type": case['image_type'],
        "image_1_base64": image1_b64,
        "image_2_base64": image2_b64
    }

    response = requests.post(f"{BASE_URL}/verify-face", json=request_data)
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"

    result = response.json()
    assert 'match' in result, "Missing 'match' in response"

    expected = case['expected_match']
    actual = result['match'] if result['status'] == 'verified' else False
    assert actual == expected, f"Expected match={expected}, got {actual} for case: {case['name']}"


def test_health():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json().get("status") in ["ok", "degraded"]


def test_version():
    response = requests.get(f"{BASE_URL}/version")
    assert response.status_code == 200
    assert "api_version" in response.json()


def test_config():
    response = requests.get(f"{BASE_URL}/config")
    assert response.status_code == 200
    assert "thresholds" in response.json()


def test_invalid_image_type():
    data = {
        "user_id": "invalid_user",
        "image_type": "wrong_type",
        "image_1_base64": "not_base64",
        "image_2_base64": "not_base64"
    }
    response = requests.post(f"{BASE_URL}/verify-face", json=data)
    assert response.status_code == 422


def test_invalid_base64():
    data = {
        "user_id": "invalid_user",
        "image_type": "id_to_selfie",
        "image_1_base64": "%%%",
        "image_2_base64": "%%%"
    }
    response = requests.post(f"{BASE_URL}/verify-face", json=data)
    assert response.status_code in [400, 422]
