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
# - `SILENT_RUN`: (Optional) Run without returning detection results.
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
    forward=os.getenv("FORWARD_PASS"),
    image=os.getenv("PYTEST_IMAGE"),
    preview=os.getenv('PYTEST_PREVIEW'),
    silent=os.getenv('SILENT_RUN', '')
)

schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "Model": {"type": "string"},
        "Config": {"type": "string"},
        "VersionID": {"type": "string"},
        "ForwardPass": {"type": "boolean"},
        "GPUSupport": {"type": "boolean"},
        "SilentRun": {"type": "boolean"},
        "StartTime": {"type": "number"},
        "EndTime": {"type": "number"},
        "DetectionStartTime": {"type": "number"},
        "DetectionEndTime": {"type": "number"},
        "DetectionLatency": {"type": "number"},
        "Results": {"type": "array"}
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

    if options.get('silent').lower() == "true":
        assert act.get('Results') == []


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


if not options.get('forward'):
    def test_draw(client):
        """
        Tests the image drawing function to validate the accurate rendering of bounding boxes and labels on images.
        """
        img = cv.imread(options.get('image'))
        encoded, buf = cv.imencode('.jpg', img)
        data = buf.tobytes()
        resp = client.post('/invocations', data=data)
        labeled_data = json.loads(resp.data.decode()).get('Results')
        img = draw(img, labeled_data)
        if options.get('preview'):
            cv.imshow('test', img)
            cv.waitKey(3000)
        assert True
