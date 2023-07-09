# Detector Test
# =============
#
# This module contains unit tests for the "detector" module, validating object detection functionality.
#
# Test Configuration:
# -------------------
# The test file requires the following environment variables to be defined:
# - `CLASSES`: (Optional) Path to the classes file.
# - `MODEL`: (Optional) Path to the model file.
# - `CONFIG`: (Optional) Path to the configuration file.
# - `FORWARD`: (Optional) Flag indicating whether to perform a forward pass and return the computation time and response.
# - `PYTEST_IMAGE`: Path to the test image.
# - `PYTEST_PREVIEW`: (Optional) flag to enable image preview during testing.
#
# Test Execution:
# ---------------
# Run with the following tests:
# - `test_ping`: Sends a GET request to the "/ping" endpoint and asserts the response status and content.
# - `test_invocations`: Sends a POST request to the "/invocations" endpoint with a test image and validates the response JSON against a schema.
# - `test_invalid_request`: Sends invalid requests to the "/invocations" endpoint to ensure appropriate error handling.
# - `test_missing_classes`: Tests the object detection service when the classes file is missing.
# - `test_draw`: Tests the image drawing function to validate the accurate rendering of bounding boxes and labels on images.
#

import json
import os

import cv2 as cv
import pytest
from jsonschema import validate

from src.detector import draw, app


@pytest.fixture
def client():
    """
    Test fixture that sets up the Flask test client for making HTTP requests to the object detection service.

    :return: Flask test client.
    """
    app.config.update({'TESTING': True})

    with app.test_client() as client:
        yield client


options = dict(
    classes=os.getenv("CLASSES"),
    model=os.getenv("MODEL"),
    config=os.getenv("CONFIG"),
    forward=os.getenv("FORWARD"),
    image=os.getenv("PYTEST_IMAGE"),
    preview=os.getenv('PYTEST_PREVIEW')
)

schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "latency": {"type": "number"},
        "output": {"type": "array"}
    }
}


def test_ping(client):
    """
    Tests the "/ping" endpoint to verify the health check functionality of the object detection service.
    """
    resp = client.get('/ping')
    assert b'{"status": "Healthy"}' in resp.data


def test_invocations(client):
    """
    Tests the "/invocations" endpoint to validate the inference functionality of the object detection service.
    """
    img = cv.imread(options.get('image'))
    encoded, buf = cv.imencode('.jpg', img)
    data = buf.tobytes()
    resp = client.post('/invocations', data=data)
    act = json.loads(resp.data.decode())
    assert validate(instance=act, schema=schema) is None


def test_invalid_request(client):
    """
    Tests an invalid request to the "/invocations" endpoint to ensure appropriate error handling.
    """
    # Send an invalid request without any data
    resp = client.post('/invocations')
    assert resp.status_code == 500

    # Send an invalid request with incorrect data format
    resp = client.post('/invocations', data=b'invalid')
    assert resp.status_code == 500

    # Set invalid values for environment vars
    os.environ['MODEL'] = '/path/invalid/model.pb'
    os.environ['CONFIG'] = '/path/invalid/config.pbtxt'

    # Send a request with invalid environment values
    resp = client.post('/invocations', data=b'dummy_data')
    assert resp.status_code == 500

    # Clean up environment variables
    os.environ.pop('MODEL', None)
    os.environ.pop('CONFIG', None)


if not options.get('forward'):
    def test_missing_classes_file(client):
        """
        Tests the object detection service when the classes file is missing.
        """
        # Set an invalid value for the CLASSES environment variable
        os.environ['CLASSES'] = '/path/to/invalid/classes.txt'

        # Load a test image
        img = cv.imread(options.get('image'))
        encoded, buf = cv.imencode('.jpg', img)
        data = buf.tobytes()

        # Send a POST request to the /invocations endpoint
        resp = client.post('/invocations', data=data)
        response_data = json.loads(resp.data.decode())

        # Assert the response status code and the presence of the 'output' field
        assert resp.status_code == 200
        assert 'output' in response_data

        # Assert the contents of the 'output' field
        for item in response_data['output']:
            assert 'id' in item
            assert 'confidence' in item
            assert 'left' in item
            assert 'top' in item
            assert 'right' in item
            assert 'bottom' in item
            assert item['id'] is not None

        # Clean up the environment variable
        os.environ.pop('CLASSES', None)


    def test_draw(client):
        """
        Tests the image drawing function to validate the accurate rendering of bounding boxes and labels on images.
        """
        img = cv.imread(options.get('image'))
        encoded, buf = cv.imencode('.jpg', img)
        data = buf.tobytes()
        resp = client.post('/invocations', data=data)
        labeled_data = json.loads(resp.data.decode()).get('output')
        img = draw(img, labeled_data)
        if options.get('preview'):
            cv.imshow('test', img)
            cv.waitKey(5000)
        assert True
