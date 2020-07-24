# Computer Pointer Controller :computer: :arrow_upper_left: 

The project serves the requirement of project submitted to Udacity in course of the Intel IoT Nanodegree. This is the third project in the compulsory projects and the fifth and last project in all the projects. It aims to create a multi-model inference pipeline that utilizes 4 models:
* **Face Detection Model**: Detects faces in the frame and returns coordinates of a bounding box around it.
   ```python3
   # Input
   Image with shape (1x3x384x672) with format #(b,c-BGR,h,w) 
   # Output
   The net outputs blob with shape: [1, 1, N, 7], where N is the number of
   detected bounding boxes. Each detection has the format 
   [image_id, label, conf, x_min, y_min, x_max, y_max] 
   ```

* **Head Pose Detection Model**: It takes in the box output of prior model and returns the inclination of the head in terms of `yaw`, `pitch` and `roll`.
```python3
# Input
    [1x3x60x60] - An input image in [1xCxHxW] format. Expected color order is BGR.

# Output   
    Output layer names in Inference Engine format:
    name: "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees).
    name: "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees).
    name: "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees). 
```

* **Facial Landmarks Regression Model**: Takes in boxed faces detected by the first model and returns coordinates of 5 facial points: left-eye, right-eye, nose, left cheek and right cheek.
```python3
# Input
    Name: "data" , shape: [1x3x48x48]
    An input image in the format [BxCxHxW] , Channels in BGR format

# Output  
    The net outputs a blob with the shape: [1, 10, 1, 1],
    containing a row-vector of 10 floating point values for five landmarks
    coordinates in the form (x0, y0, x1, y1, ..., x5, y5).
    All the coordinates are normalized to be in range [0,1].
```

* **Gaze Estimation Model**: The model uses the the coordinates of the boxes bounding the left-eye and right-eye; we obtain them by manipulating the facial-landmarks for them. The head inclinations are obtained by the head position model.
```python3
# Input
    Blob left_eye_image and the shape [1x3x60x60] in the format [BxCxHxW]
    Blob right_eye_image and the shape [1x3x60x60] in the format [BxCxHxW]
    Blob head_pose_angles and the shape [1x3] in the format [BxC]

# Output
    The net outputs a blob "gaze_vector" with the shape: [1, 3], containing
        Cartesian coordinates of gaze direction vector.
        The output vector is not normalized and has non-unit length.
```

The `x`, `y` and `z` coordinates for the gaze are then fed to the `mouse_controller` module that utilizes `pyautogui` package to control the mouse pointer. 


## Project Set Up and Installation
The project is setup as follows:
```bash
.
├── bin
│   ├── demo.mp4
│   ├── img.png
│   └── mouse_pointer1.gif
├── landmarks_result.png
├── main.py
├── models
│   └── intel
│       ├── face-detection-adas-binary-0001
│       │   └── FP32-INT1
│       │       ├── face-detection-adas-binary-0001.bin
│       │       └── face-detection-adas-binary-0001.xml
│       ├── gaze-estimation-adas-0002
│       │   ├── FP16
│       │   │   ├── gaze-estimation-adas-0002.bin
│       │   │   └── gaze-estimation-adas-0002.xml
│       │   ├── FP32
│       │   │   ├── gaze-estimation-adas-0002.bin
│       │   │   └── gaze-estimation-adas-0002.xml
│       │   └── FP32-INT8
│       │       ├── gaze-estimation-adas-0002.bin
│       │       └── gaze-estimation-adas-0002.xml
│       ├── head-pose-estimation-adas-0001
│       │   ├── FP16
│       │   │   ├── head-pose-estimation-adas-0001.bin
│       │   │   └── head-pose-estimation-adas-0001.xml
│       │   ├── FP32
│       │   │   ├── head-pose-estimation-adas-0001.bin
│       │   │   └── head-pose-estimation-adas-0001.xml
│       │   └── FP32-INT8
│       │       ├── head-pose-estimation-adas-0001.bin
│       │       └── head-pose-estimation-adas-0001.xml
│       └── landmarks-regression-retail-0009
│           ├── FP16
│           │   ├── landmarks-regression-retail-0009.bin
│           │   └── landmarks-regression-retail-0009.xml
│           ├── FP32
│           │   ├── landmarks-regression-retail-0009.bin
│           │   └── landmarks-regression-retail-0009.xml
│           └── FP32-INT8
│               ├── landmarks-regression-retail-0009.bin
│               └── landmarks-regression-retail-0009.xml
├── out.avi
├── README.md
├── requirements.txt
├── res1.png
├── res576.png
├── res.png
├── src
│   ├── face_detection.py
│   ├── facial_landmarks_detection.py
│   ├── gaze_detection.py
│   ├── head_pose_estimation.py
│   ├── input_feeder.py
│   ├── mouse_controller.py
│   └── __pycache__
│       ├── face_detection.cpython-36.pyc
│       ├── facial_landmarks_detection.cpython-36.pyc
│       ├── gaze_detection.cpython-36.pyc
│       ├── head_pose_estimation.cpython-36.pyc
│       ├── input_feeder.cpython-36.pyc
│       └── mouse_controller.cpython-36.pyc
├── starter.zip
└── start.sh
```
#### 1. Activate your virtual environment and Setup OpenVINO environment Variables
I prefer doing it using a `bash` file (__start.sh__ in my case), defined as
```bash
#!/bin/bash
source ~/.virtualenvs/cv/bin/activate
source /opt/intel/openvino/bin/setupvars.sh
``` 

where `cv` is my virtual environment. Now, source it
```bash
$ source start.sh
```


#### 2. Install Dependencies: 
Use the `requirements.txt` to install. 
```bash
# cd into the ComputerPointerController directory
$ pip install -r requirements.txt
```
 
#### 3. Download Models:
For a fresh start, models need to be downloaded and placed into the `models` directory as shown above. It is recommended to use the `model_downloader.py` in the directory `/opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/`. To download models  from the model zoo. 
```bash
$ python /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name face-detection-adas-binary-0001 -o /models/

$ python /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name head-pose-estimation-adas-0001 -o /models/

$ python /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name landmarks-regression-retail-0009 -o /models/

$ python /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name gaze-estimation-adas-0002 -o /models/
```

## Demo
The `main.py` module handles all the other modules. The application can be run as 
```bash
$ python3 main.py -l 0 -fd_path models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 -hpe_path models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 -fld_path models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009 -gd_path models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002 -fd_th 0.3 -i bin/img.png
```

On successful execution, you should get something like this
![execution1](https://github.com/pra-dan/Intel-EdgeAI-Nanodegree/blob/master/ComputerPointerController/bin/1.png)


## Documentation
The run can be controlled by knowing the arguments. 
```bash
usage: main.py [-h] [-l LOG_FLAG] -fd_path FACE_DETECTION_MODEL_PATH -hpe_path
               HEAD_POSE_ESTIMATION_MODEL_PATH -fld_path
               FACIAL_LANDMARKS_DETECTION_MODEL_PATH -gd_path
               GAZE_DETECTION_MODEL_PATH [-fd_th FACE_DETECTION_THRESHOLD] -i
               INPUT [-d DEVICE]

optional arguments:
  -h, --help            show this help message and exit
  -l LOG_FLAG, --log_flag LOG_FLAG
                        True: Log | False: No Logs
  -fd_path FACE_DETECTION_MODEL_PATH, --face_detection_model_path FACE_DETECTION_MODEL_PATH
                        Path to an xml file face_detection model.
  -hpe_path HEAD_POSE_ESTIMATION_MODEL_PATH, --head_pose_estimation_model_path HEAD_POSE_ESTIMATION_MODEL_PATH
                        Path to an xml file head_pose_estimation model.
  -fld_path FACIAL_LANDMARKS_DETECTION_MODEL_PATH, --facial_landmarks_detection_model_path FACIAL_LANDMARKS_DETECTION_MODEL_PATH
                        Path to an xml file facial_landmarks_detection model.
  -gd_path GAZE_DETECTION_MODEL_PATH, --gaze_detection_model_path GAZE_DETECTION_MODEL_PATH
                        Path to an xml file gaze_detection_detection model.
  -fd_th FACE_DETECTION_THRESHOLD, --face_detection_threshold FACE_DETECTION_THRESHOLD
```

To give the user control over the logging, I have added an argument to disable all flags. The default disables any extra log. On enabling it, we get results as
![execution2](https://github.com/pra-dan/Intel-EdgeAI-Nanodegree/blob/master/ComputerPointerController/bin/2.png)

Further debug level logs can be triggered by changing the `logConfig` from 
```python3
log.basicConfig(format='[INFO] \t %(message)s', level=log.INFO)
``` 
to
```python3
log.basicConfig(format='[INFO] \t %(message)s', level=log.DEBUG)
```

## Benchmarks
```bash
# Model Loading Time (in seconds)
                                 INT8       FP16       FP32 
Face Detection               |    -     |    -     |  0.6068  |
Head Pose Estimation         |  2.1641  |  0.3111  |  0.4235  |
Facial Landmarks Detection   |  0.4974  |  0.2040  |  0.2497  |
Gaze Detection               |  2.0417  |  0.3693  |  0.5139  |

# Model Inference Time per frame
                                 INT8       FP16       FP32 
Face Detection               |    -     |    -     |  0.0039  |
Head Pose Estimation         |  0.0002  |  0.0030  |  0.0002  |
Facial Landmarks Detection   |  0.0002  |  0.0001  |  0.0002  |
Gaze Detection               |  0.0053  |  0.0004  |  0.0004  |
```

## Results
* The Face Detection Model was only available with a single precision: FP32.
* On looking at the network architectures of the Gaze Detection for all three model precisions, (in order FP32-INT8 -> FP16 -> FP32)

![architectures](https://github.com/pra-dan/Intel-EdgeAI-Nanodegree/blob/master/ComputerPointerController/bin/precision_net.gif)

We notice that the INT8 precision model has additional layers `FakeQuantize` that are responsible for quantising the model to INT8 precision initially FP32. 

* A successful face detection looks like this
![result_face](https://github.com/pra-dan/Intel-EdgeAI-Nanodegree/blob/master/ComputerPointerController/bin/res.png) 

* A successful execution of all three models (Face Detection, Facial Landmarks Detection, Head Pose Estimation) looks this way
![result_all](https://github.com/pra-dan/Intel-EdgeAI-Nanodegree/blob/master/ComputerPointerController/bin/landmarks_result.png) 

* With some latency, the desired result looks as shown below. 

![result](https://github.com/pra-dan/Intel-EdgeAI-Nanodegree/blob/master/ComputerPointerController/bin/mouse_pointer1.gif) 

__Note: I had to drag the pointer inside the frame, whenever it escaped outside.__
___
## Edge Cases
Some edge cases observed are:
#### Mouse Pointer Moving to Edge of Screen:
The issue is an inherent property of the `pyautogui` package and is triggered as soon as the computer-contorlled-mouse reaches the edge of the screen. It gives an error as:
>  pyautogui.FailSafeException: PyAutoGUI fail-safe triggered
   from mouse moving to a corner of the screen. To disable this fail-safe,
   set pyautogui.FAILSAFE to False. 
This is not recommended but leads to complete freedom for the mouse pointer. 

#### True Negatives:
In case of such occurance, for each of the model, a dummy result is used to prevent the pipeline from getting blocked/stopped. This happens in case of __True Negatives__.
```python3
yaw, pitch, roll, x, y = 0,0,0
if(hpe.wait() == 0):
    yaw, pitch, roll = hpe.preprocess_output()
```

To troubleshoot the problem, logs can be used as

```python3
try:
    return face, x,y,z
except:
    log.error(f"!!! The Model {self.model_name} could not find any Gazing Person")
```

#### No Face detected:
When a face is not detected, due to absence of person or model inability, the application does not break down and continues to work with the available results. 

![wrong_result](https://github.com/pra-dan/Intel-EdgeAI-Nanodegree/blob/master/ComputerPointerController/bin/res576.png) 

This has the disadvantage that the mouse pointer moves unintentionally.
