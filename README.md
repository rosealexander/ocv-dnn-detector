# OpenCV Deep Neural Network Detector

This is a Flask application for performing object detection. Supports using models in formats such as caffemodel, pb, t7, net, weights, bin, and onnx, and leverages the capabilities of [OpenCV DNN module](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html) for efficient and accurate detection.

## Overview

The Flask application provides two routes:

- `/ping`: Handles health checks and returns a JSON response indicating the health status.
- `/invocations`: Handles inference requests and returns a JSON response containing the inference results.

## Getting Started

### Prerequisites
Before running the application, make sure you have the following prerequisites:

- Python >= **3.9**

- Required dependencies [see requirements.txt](requirements.txt)

### Set environment variables:

  - **MODEL**: (optional) Path to the model file.
  - **CONFIG**: (optional) Path to the configuration file.
  - **CLASSES**: (optional) Path to the file containing class names.
  - **GPU_SUPPORT**: (default: False) Boolean indicating whether GPU support is enabled.
  - **FORWARD_PASS**: (default: False) Boolean indicating response output from [cv.dnn.Net.forward](https://docs.opencv.org/3.4/db/d30/classcv_1_1dnn_1_1Net.html#a00e707a86b2da4f980f9342b1fc2cc92).
  - **SILENT_RUN**: (optional) Run detector without returning object detection results.
  - **CONFIDENCE_THRESHOLD**: (default: 0.5) Confidence threshold for object detection.
  - **NMS_THRESHOLD**: (default: 0.5) Non-Maximum Suppression (NMS) threshold for object detection.
  - **MODEL_SERVER_WORKERS**: (default: auto) Number of workers, defaults to number of cpu cores.
  - **MODEL_SERVER_TIMEOUT**: (default: 60) Server timout in seconds.
  - **LOG_LEVEL**: (default: WARNING) The application log level.

Note: If the **MODEL** or **CONFIG** are not defined, their values will be automatically determined based on predefined file extension patterns and files available in the current directory and its subdirectories.

### Start the Development server:

```shell
bin/start-dev
```

## API Routes

### Health Check

#### Endpoint: /ping

Used for health checks and returns a JSON response indicating the health status of the application.

Example Response:

```json
{
  "status": "Healthy"
}
```

### Object Detection

#### Endpoint: /invocations

Used for performing object detection on an input image. It expects a POST request with image data in the request body. Its response contains the inference results, including the latency, in seconds, and labeled objects.

Example Request:

```shell
curl -X POST -H "Content-Type: image/jpeg" --data-binary "@image.jpg" <application_url>/invocations
```

Example Response
```json
{
  "Model": "model.pb",
  "Config": "config.cfg",
  "VersionID": 1,
  "Forward": false,
  "GPUSupport": true,
  "SilentRun": false,
  "StartTime": 1632833372.123456,
  "EndTime": 1632833372.987654,
  "DetectionStartTime": 1632833372.456789,
  "DetectionEndTime": 1632833372.987654,
  "DetectionLatency": 0.5312,
  "DetectionResults": [
    {
      "id": "person",
      "confidence": 0.85,
      "left": 120,
      "top": 150,
      "right": 320,
      "bottom": 480
    },
    {
      "id": "car",
      "confidence": 0.92,
      "left": 400,
      "top": 200,
      "right": 650,
      "bottom": 450
    }
  ]
}
```

#### Response Structure

- **Model** The model filename used for detection.
- **Config** The config filename used for detection.
- **Version** The current ocv-dnn-detector application version.
- **ForwardPass** If response output format is [cv.dnn.Net.forward](https://docs.opencv.org/3.4/db/d30/classcv_1_1dnn_1_1Net.html#a00e707a86b2da4f980f9342b1fc2cc92).
- **GPUSupport** If GPU support is enabled.
- **SilentRun** If detector ran without returning object detection results.
- **StartTime** The response start time.
- **EndTime** The response end time
- **DetectionStartTime** The object detection start time.
- **DetectionEndTime** The object detection end time.
- **DetectionLatency** The object detection latency.
- **Results** The object detection results.

## Model Zoo

The following models have been tested to work:

| Model                                                        | Weights File                                                                                                                                       | Config File                                                                                                                                                          |
|--------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Faster-RCNN Inception v2](docs/Faster-RCNN-Inception-v2.md) | [Download](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)                                 | [Download](https://github.com/opencv/opencv_extra/raw/4.x/testdata/dnn/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt)                                               |
| [Mask-RCNN Inception v2](docs/Mask-RCNN-Inception-v2.md)     | [Download](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz)                                   | [Download](https://github.com/opencv/opencv_extra/raw/4.x/testdata/dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt)                                                 |
| [MobileNet-SSD v3 Large](docs/MobileNet-SSD-v3-Large.md)     | [Download](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz)                                   | [Download](https://gist.github.com/dkurt/54a8e8b51beb3bd3f770b79e56927bd7/raw/2a20064a9d33b893dd95d2567da126d0ecd03e85/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt) |
| [YOLOv4](docs/yolov4.md)                                     | [Download](https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights)                                                            | [Download](https://github.com/AlexeyAB/darknet/raw/master/cfg/yolov4.cfg)                                                                                            |

* Please note that these are provided as examples and additional models may be integrated.

## License

Copyright 2023 Alexander Rose. All Rights Reserved.

This project is licensed under the Apache License, Version 2.0. See the LICENSE file for more information.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
