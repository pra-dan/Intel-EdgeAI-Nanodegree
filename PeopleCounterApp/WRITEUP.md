# Project Write-Up

## Running the Project

### Step 1 - Start the Mosca server

```
cd webservice/server/node-server
node ./server.js
```

You should see the following message, if successful:
```
Mosca server started.
```

### Step 2 - Start the GUI

Open new terminal and run below commands.
```
cd webservice/ui
npm run dev
```

You should see the following message in the terminal.
```
...
Project is running at http://0.0.0.0:3000/
...
...
webpack: Compiled successfully
```
**NOTE: This address is where our project is running. It may vary on your configuration. So, use the same address that gets printed as shown above.**
Open a tab in your browser and enter the address, in my case: http://0.0.0.0:3000/

### Step 3 - FFmpeg Server

Open new terminal and run the below commands.
```
sudo ffserver -f ./ffmpeg/server.conf
```
You should see the message like:
```
Tue May 12 16:34:17 2020 FFserver started.
```
### Step 4 - Setup the OpenVINO environmet
* Open a new terminal and source the environmet bash script (if OpenVINO environmet is not automatically sourced using `.bashrc`)
```
source start.sh
```
* (Conditional) Launch OpenCV virtual environmet ONLY if you have installed OpenCV in a virtual environmet. Otherwise, you are allowed to skip this step.
```
workon cv
# cv is the name of my v-environmet
```

### Step 5 - Run the code

The Run command and arguments should be like:
```
$ python3 main.py -m popo_models/mystic_frozen_darknet_yolov3_tiny_model.xml -i /opt/intel/openvino_2020.1.023/Intel-EdgeAI-Nanodegree/PeopleCounterApp/resources/pedes_detect.mp4 -pt 0.3 -cl coco.names | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

## Project Notes:
* To log, uncomment the logging imports
```
# uncomment for logging
#import logging
#logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
#log = logging.getLogger()
```
in `inference.py` and `main.py` files. This is done to reduce this overload and focus computation on the real-time detection.

* The webcam input feature is not tested due to an implicit OpenCV issue reported by me [here](https://github.com/opencv/opencv/issues/17221).

* The project has been completed in various phases, with each phase saved as a branch of this GitHub project. The different progressive branches and their checkpoints are stated below:

Branch Name | Checkpoint completed
--|--
b_nicely_working_v3tiny | Test inference made by the converted YOLOv3-tiny model
b_done_finding_stats | Finish calculating the statistics of `current_count`, `total_count` and `duration` and test locally
b_server_set_up_project_done | Send the stats and frame to the respective servers and finish with the Project

## Explaining Custom Layers

The YOLO model was free of any unsupported sub-graphs or layers, hence, no custom layers were implemented.

## Model Research
* Converted the Darknet based YOLOv3-tiny model using [Mystic's method](https://github.com/mystic123/tensorflow-yolo-v3.git)
```%shell%
# Install TensorFlow
!pip install tensorflow==1.11.0

# Get repository
!git clone https://github.com/mystic123/tensorflow-yolo-v3.git
%cd tensorflow-yolo-v3
!git checkout ed60b90

# Get coco.names
!wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

# Get v3-tiny weights
!wget "https://pjreddie.com/media/files/yolov3-tiny.weights"

# Convert v3-tiny
!python3 convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3-tiny.weights --tiny
```
* Converting TensorFlow graph file (.pb) to IR information
```%shell%
pradan@pradan-HP-15-Notebook-PC:/opt/intel/openvino_2020.1.023/deployment_tools/model_optimizer$ python3 mo_tf.py --input_model ~/yolo-models/mystic_frozen_darknet_yolov3_tiny_model.pb --tensorflow_use_custom_operations_config extensions/front/tf/yolo_v3_tiny.json --batch 1 --output_dir ~/Desktop

```
The conversion was successfully executed, as shown in the log:
```%shell%
[ SUCCESS ] Generated IR version 10 model.
[ SUCCESS ] XML file: /home/pradan/Desktop/mystic_frozen_darknet_yolov3_tiny_model.xml
[ SUCCESS ] BIN file: /home/pradan/Desktop/mystic_frozen_darknet_yolov3_tiny_model.bin
[ SUCCESS ] Total execution time: 58.47 seconds.
[ SUCCESS ] Memory consumed: 516 MB.
```

## Comparing Model Performance

I have used the [YOLO v3-tiny model](https://pjreddie.com/media/files/papers/YOLOv3.pdf), based on Darknet. My method(s) to compare models before and after conversion to Intermediate Representations
were as follows:
Parameter|Pre-Conversion|Post-Conversion
--|--|--
Size|33.79 MB | 35.4 MB
Average Inference Time|1.90 s| 0.003 s
Accuracy of detecting class 'Dog'|57% | 66%
Peak CPU Overhead|97.5% |38.8%

Although there was an impressive boost in the execution time, but this was achieved at the cost of lower accuracy; many classes were not detected. This also justifies for the increase in _Dog_ class detection. The results can be seen as follows:

![pre-conversion](https://github.com/PrashantDandriyal/Intel-EdgeAI-Nanodegree/blob/b_server_set_up_project_done/PeopleCounterApp/resources/results_pre.jpg)
![pre-conversion](https://github.com/PrashantDandriyal/Intel-EdgeAI-Nanodegree/blob/b_server_set_up_project_done/PeopleCounterApp/resources/results_post.jpg)

The CPU usage is shown below:   
![pre-conversion](https://github.com/PrashantDandriyal/Intel-EdgeAI-Nanodegree/blob/b_server_set_up_project_done/PeopleCounterApp/resources/cpu_usage_default.gif)
![pre-conversion](https://github.com/PrashantDandriyal/Intel-EdgeAI-Nanodegree/blob/b_server_set_up_project_done/PeopleCounterApp/resources/cpu_usage_openvino.gif)

## Assess Model Use Cases

Some of the potential use cases of the people counter app are:
* **Social Distancing Monitoring System:** This application suits the recently needed social distancing measures, perfectly. It can be deployed at the shopping stores and common places that see multiple people gathering at a place. Hardware can be edge devices like the CCTV cameras or as a small addon to it, that ensures that there are not more than a particular number of people in the store and alarms can be raised if the number exceeds.

* **Buyer Time Monitoring System:** The application also suits another related purpose: the need to monitor if all the costomers/buyers get equal or rather not more than the specified time, thus ensuring that no buyer stays in the store for more than time T and others can get to buy the commodities without having to wait for long in wearsome queues. If the average time can be fixed and shared with the customers, then each customer can know exactly how long does he/she have to wait before they get the chance (knowing how many people are standing in front of them at that instant).

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

### References
* [OpenVINO Toolkit API Classes](https://docs.openvinotoolkit.org/2019_R3/ie_python_api.html)
