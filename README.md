# OpenCV Deep Neural Network Detector

This is a Flask application, suitable for use with [AWS SageMaker hosting services](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html), for performing object detection using models in formats such as caffemodel, pb, t7, net, weights, bin, and onnx, leveraging the capabilities of OpenCV DNN module for efficient and accurate detection.

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

  - **MODEL**: Path to the model file (optional).
  - **CONFIG**: Path to the configuration file (optional).
  - **CLASSES**: Path to the file containing class names (optional).
  - **GPU_SUPPORT**: Flag indicating whether GPU support is enabled (default: False).
  - **FORWARD**: Flag indicating whether to perform a forward pass and return the computation time and response (default: False).
  - **CONFIDENCE_THRESHOLD**: Confidence threshold for object detection (default: 0.1).
  - **NMS_THRESHOLD**: Non-Maximum Suppression (NMS) threshold for object detection (default: 0.5).
  - **MODEL_SERVER_WORKERS**: Number of workers (default: the number of CPU cores).
  - **MODEL_SERVER_TIMEOUT**: Server timout in seconds (default: 60).

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
  "latency": 0.5623,
  "output": [
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

## Model Zoo

The following models have been tested to work:

| Model                                                        | Weights File                                                                                                                                       | Config File                                                                                                                                                          |
|--------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Faster-RCNN Inception v2](docs/Faster-RCNN-Inception-v2.md) | [Download](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)                                 | [Download](https://github.com/opencv/opencv_extra/raw/4.x/testdata/dnn/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt)                                               |
| [Mask-RCNN Inception v2](docs/Mask-RCNN-Inception-v2.md)     | [Download](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz)                                   | [Download](https://github.com/opencv/opencv_extra/raw/4.x/testdata/dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt)                                                 |
| [MobileNet-SSD v3](docs/MobileNet-SSD-v3.md)                 | [Download](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz)                                   | [Download](https://gist.github.com/dkurt/54a8e8b51beb3bd3f770b79e56927bd7/raw/2a20064a9d33b893dd95d2567da126d0ecd03e85/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt) |
| [YOLOv4](docs/yolov4.md)                                     | [Download](https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights)                                                            | [Download](https://github.com/AlexeyAB/darknet/raw/master/cfg/yolov4.cfg)                                                                                            |

* Please note that these are provided as examples and additional models may be integrated.

## Deployment

This containerized application is ready to deploy on AWS SageMaker hosting services, refer to the [AWS SageMaker documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html).

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
