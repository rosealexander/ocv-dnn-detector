#
# Copyright 2023 Alexander Rose. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This is a Flask application for performing object detection using models in formats such as caffemodel, pb, t7, net,
# weights, bin, and onnx, leveraging the capabilities of OpenCV DNN for efficient and accurate detection.
#
# Overview:
# ----------------
# The module defines a Flask application with two routes:
# - `/ping`: Handles health checks and returns a JSON response indicating the health status.
# - `/invocations`: Handles inference requests and returns a JSON response containing the inference results.
#
# Functions:
# ----------
# - ping(): Handles the `/ping` endpoint for health checks.
# - invocations(): Handles the `/invocations` endpoint for inference.
# - detect(model: str, config: str, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]: Performs object detection on an image using the specified model and configuration.
# - label(res: tuple) -> list[dict]: Generates labeled objects from the detection results.
# - infer(data: bytes, model_path: str | os.PathLike, config_path: str | os.PathLike, names_path: str | os.PathLike = None) -> list[dict]: Performs inference on input data using the specified model.
# - draw(img: np.ndarray, labeled_data: list[dict]) -> np.ndarray: Draws bounding boxes and labels on the input image.
#
# Note:
# -----
# The module allows the following environment variables:
#
# MODEL: (Optional) Path to the model file.
# CONFIG: (Optional) Path to the configuration file.
# CLASSES: (Optional) Path to the classes file.
# FORWARD: (default: False) Boolean indicating whether to return results from cv.dnn.Net.forward rather than cv.dnn.DetectionModel.detect.
# GPU_SUPPORT: (default: False) Boolean indicating whether GPU support is enabled.
# SILENT_RUN: (default: False) Boolean to run detector without returning object detection results.
# CONFIDENCE_THRESHOLD (default: 0.5) Confidence threshold for object detection.
# NMS_THRESHOLD: (default: 0.5) Non-Maximum Suppression (NMS) threshold for object detection.
# IMAGE_SCALE_FACTOR: (optional) The input scalefactor value to match neural network model requirements, value is inferred if none provided.
# MODEL_SERVER_WORKERS: (default: auto) Number of workers, defaults to number of cpu cores.
# MODEL_SERVER_TIMEOUT: (default: 60) Server timout in seconds.
# LOG_LEVEL: (default: WARNING) The application logging level.
#
# If MODEL or CONFIG is not defined, the module automatically determines their values based on predefined file extension
# patterns and the files available in the current directory and its subdirectories.
#

import glob
import json
import logging
import os
import sys
import re
import time
import timeit
import warnings
from typing import Any, Sequence

import cv2 as cv
import flask
import io
from PIL import Image
import numpy as np

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from version import __version__

VERSION = __version__

with warnings.catch_warnings():
    # Ignore deprecation and future warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

# Capture return value with the timeit module
timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        retval = {stmt}
    _t1 = _timer()
    return _t1 - _t0, retval
"""

# Read environment variables
MODEL = os.getenv('MODEL', None)
CONFIG = os.getenv('CONFIG', None)
CLASSES = os.getenv('CLASSES', None)
FORWARD_PASS = os.getenv('FORWARD_PASS', '').lower() == 'true'
GPU_SUPPORT = os.getenv('GPU_SUPPORT', '').lower() == 'true'
SILENT_RUN = os.getenv('SILENT_RUN', '').lower() == 'true'

CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.5))
NMS_THRESHOLD = float(os.getenv('NMS_THRESHOLD', 0.5))
LOG_LEVEL = os.getenv('LOG_LEVEL', 'WARNING').upper()
SCALE_FACTOR = os.getenv('SCALE_FACTOR', 1.0)
DEBUG = os.getenv('DEBUG', False)

# Configure logging
logger = logging.getLogger(__name__)

if LOG_LEVEL == 'DEBUG' or DEBUG:
    LOG_LEVEL = 'DEBUG'
    logger.setLevel(level=logging.DEBUG)
elif LOG_LEVEL == 'INFO':
    logger.setLevel(level=logging.INFO)
elif LOG_LEVEL == 'ERROR':
    logger.setLevel(level=logging.ERROR)
elif LOG_LEVEL == 'CRITICAL':
    logger.setLevel(level=logging.CRITICAL)
else:
    logger.setLevel(level=logging.WARNING)

# Configure MODEL and CONFIG
if MODEL is None or CONFIG is None:
    """
    Check if MODEL and CONFIG are already defined. If not, perform the following steps:
    
    1. Define patterns to match desired file extensions.
    2. Get all files in the current directory and subdirectories.
    3. Filter files based on the file extensions.
    4. Find the first matching model file.
    5. Check the file extension of the MODEL and find the corresponding configuration file.
    6. Check if CONFIG is None, unless MODEL ends with .t7, .net, or .onnx.
    """

    model_pattern = r"\.(caffemodel|pb|t7|net|weights|bin|onnx)$"
    config_pattern = r"\.(prototxt|pbtxt|cfg|xml)$"
    files = [file for file in glob.glob('**/*', recursive=True) if not file.startswith(('.', 'bin/', 'docs/', 'src/', 'tests/', 'venv/'))]
    config_files = [file for file in files if re.search(config_pattern, file)]
    model_files = [file for file in files if re.search(model_pattern, file)]

    MODEL = MODEL or next((c for c in model_files), None)

    if MODEL is None:
        raise ValueError("MODEL cannot be None")
    elif MODEL.endswith('.caffemodel'):
        deploy_prototxt = next((c for c in config_files if c.endswith('deploy.prototxt')), None)
        CONFIG = CONFIG or deploy_prototxt or next((c for c in config_files if c.endswith('.prototxt')), None)
    elif MODEL.endswith('.pb'):
        CONFIG = CONFIG or next((c for c in config_files if c.endswith('.pbtxt')), None)
    elif MODEL.endswith('.weights'):
        CONFIG = CONFIG or next((c for c in config_files if c.endswith('.cfg')), None)
        if "yolo" in MODEL.lower():
            SCALE_FACTOR = os.getenv('SCALE_FACTOR', (1.0 / 255))
    elif MODEL.endswith('.bin'):
        CONFIG = CONFIG or next((c for c in config_files if c.endswith('.xml')), None)

    if CONFIG is None and not (
            MODEL.endswith('.t7') or MODEL.endswith('.net') or MODEL.endswith('.onnx') or MODEL.endswith('.pb')):
        raise ValueError("CONFIG cannot be None")


def forward(model: str, config: str, img: np.ndarray) -> Sequence[Any]:
    height, width, _ = img.shape

    # Load the model
    net = cv.dnn.readNet(model, config)

    net.setInput(cv.dnn.blobFromImage(img, scalefactor=SCALE_FACTOR, size=(width, height), swapRB=True, crop=False))

    if GPU_SUPPORT:
        # Set preferable backend and target for GPU support
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

    return net.forward(net.getUnconnectedOutLayersNames())


def detect(model: str, config: str, img: np.ndarray) -> tuple[np.ndarray]:
    """
    Perform object detection on an image using the specified model and configuration.

    :param model: Path to the model file.
    :type model: str
    :param config: Path to the configuration file.
    :type config: str
    :param img: Image on which to perform object detection.
    :type img: numpy.ndarray
    :return: Detected objects.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    height, width, _ = img.shape

    # Load the model
    net = cv.dnn.readNet(model, config)

    if GPU_SUPPORT:
        # Set preferable backend and target for GPU support
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

    # Create the detection model
    detection_model = cv.dnn_DetectionModel(net)
    detection_model.setInputSize(width, height)
    detection_model.setInputScale(SCALE_FACTOR)
    detection_model.setInputSwapRB(True)

    # Perform object detection
    return detection_model.detect(img, **dict(
        confThreshold=CONFIDENCE_THRESHOLD,
        nmsThreshold=NMS_THRESHOLD
    ))


def label(res: tuple) -> list[dict]:
    """
    Generate labeled objects from detection results.

    :param res: Detection results as a tuple containing class IDs, confidences, and bounding boxes.
    :return: Labeled objects as a list of dictionaries with ID, confidence, and bounding box coordinates.
    """
    try:
        class_ids, confidences, boxes = res
        return [
            dict(
                id=res[0],
                confidence=res[1],
                left=int(res[2][0]),
                top=int(res[2][1]),
                right=int(res[2][0] + res[2][2]),
                bottom=int(res[2][1] + res[2][3]),
            ) for res in list(zip(class_ids.tolist(), confidences.tolist(), boxes.tolist()))
        ]
    except AttributeError:
        return []


def infer(data: bytes, model_path: str, config_path: str, names_path: str = None, **kwargs) -> dict:
    """
    Performs object detection and returns the inference computation time and labeled predictions.

    :param data: Input data as bytes representing an image.
    :type data: bytes
    :param model_path: Path to the model file.
    :type model_path: str | os.PathLike
    :param config_path: Path to the configuration file.
    :type config_path: str | os.PathLike
    :param names_path: Optional path to the file containing class names.
    :type names_path: str | os.PathLike, optional
    :param kwargs: Additional keyword arguments.
    :type kwargs: dict, optional
    Keyword Arguments:
        * forward (bool): Optional flag indicating whether to perform a forward pass and return the computation time and response.
    :return: A dictionary containing inference statistics and labeled objects.
    :rtype: dict
    :warns: FileNotFoundError: If the `names_path` file is not found.

    :Example:

    >>> image = cv.imread('image')
    >>> encoded, buf = cv.imencode('.jpg', img)
    >>> img_data = buf.tobytes()
    >>> model = 'model.pb'
    >>> config = 'config.cfg'
    >>> classes = 'class_names.txt'
    >>> result = infer(img_data, model, config, classes)
    {'detection_latency': 0.5623, 'start_time': 1632833372.123456, 'end_time': 1632833372.987654, 'results': [{'id': 'person', 'confidence': 0.85}, {'id': 'car', 'confidence': 0.92}]}
    >>> result = infer(img_data, model, config, forward=True)
    {'detection_latency': 0.3412, 'start_time': 1632833372.123456, 'end_time': 1632833372.987654, 'results': <forward_result>}


    """
    img = cv.cvtColor(np.array(Image.open(io.BytesIO(data))), cv.COLOR_BGR2RGB)

    if kwargs.get('forward', False):
        t0 = time.time()
        l, resp = timeit.timeit(lambda: forward(model_path, config_path, img), number=1)
        t1 = time.time()
        results = tuple([ctx.tolist() for ctx in resp])
        return dict(latency=l, start_time=t0, end_time=t1, results=results)

    t0 = time.time()
    l, resp = timeit.timeit(lambda: detect(model_path, config_path, img), number=1)
    t1 = time.time()
    results = label(resp)

    try:
        if names_path:
            with open(names_path, 'r') as fp:
                names = fp.read().rstrip("\n").split("\n")
                for lab in results:
                    lab.update(dict(id=names[lab.get("id")]))
    except FileNotFoundError as e:
        logger.warning(e)
        pass

    return dict(latency=l, start_time=t0, end_time=t1, results=results)


def draw(img: np.ndarray, labeled_data: list[dict]) -> np.ndarray:
    """
    Draw bounding boxes and labels on the input image.

    :param img: Input image as a NumPy array.
    :param labeled_data: Labeled objects as a list of dictionaries.
    :return: Image with bounding boxes and labels drawn.
    """
    for lab in labeled_data:
        start_point = (lab.get('left'), lab.get('top'))
        end_point = (lab.get('right'), lab.get('bottom'))
        img = cv.rectangle(img, start_point, end_point, (255, 0, 0), 2)
        point = (lab.get('left'), lab.get('top') - 10)
        cv.putText(img, str(lab.get('id')), point, cv.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
    return img


app = flask.Flask(__name__)
app.logger.setLevel(logger.level)


@app.route('/ping', methods=['GET'])
def ping() -> flask.Response:
    """
    Handle the ping endpoint for health checks.

    :return: Flask Response object with a JSON response indicating the health status.
    """
    status = 200
    return flask.Response(
        response=json.dumps(dict(status="Healthy")),
        status=status,
        mimetype='application/json'
    )


@app.route("/invocations", methods=["POST"])
def invocations() -> flask.Response:
    """
    Handle the invocations endpoint for inference.

    :return: Flask Response object with a JSON response containing the inference results.
    """
    t0 = time.time()

    flask_response = dict(
        response=json.dumps(dict()),
        status=None,
        mimetype='application/json'
    )

    try:
        data = flask.request.data

        resp = infer(data, MODEL, CONFIG, CLASSES, forward=FORWARD_PASS)

        detection_start_time = resp.get('start_time')
        detection_end_time = resp.get('end_time')
        detection_latency = resp.get('latency')

        if SILENT_RUN:
            detection_results = []
        else:
            detection_results = resp.get('results')

        t1 = time.time()

        flask_response.update(
            response=json.dumps(
                dict(
                    Model=os.path.basename(MODEL),
                    Config=os.path.basename(CONFIG),
                    VersionID=VERSION,
                    ForwardPass=FORWARD_PASS,
                    GPUSupport=GPU_SUPPORT,
                    SilentRun=SILENT_RUN,
                    StartTime=t0,
                    EndTime=t1,
                    DetectionStartTime=detection_start_time,
                    DetectionEndTime=detection_end_time,
                    DetectionLatency=detection_latency,
                    Results=detection_results,
                )
            ),
            status=200,
            mimetype='application/json'
        )
    except Exception as ex:
        logger.error(ex, exc_info=True)
        flask_response.update(
            response=None,
            status=500,
        )
    finally:
        return flask.Response(**flask_response)
