Run using
python3 main.py -m popo_models/yolo-v2-coco.xml -i images/sample_dog.jpg -d CPU -pt 0.5

# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

The process behind converting custom layers involves...

Some of the potential reasons for handling custom layers are...

## Comparing Model Performance

I have used the [YOLO v3-tiny model](https://pjreddie.com/media/files/papers/YOLOv3.pdf), based on Darknet. My method(s) to compare models before and after conversion to Intermediate Representations
were as follows:
Parameter|Pre-Conversion|Post-Conversion
--|--|--
Size|33.79 MB | 35.4 MB
Total Execution Time|43.34 s| 2.67 s

Although there was an impressive boost in the exeecution time, but this was achieved at the cost of lower accuracy; many classes were not detected.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

Each of these use cases would be useful because...

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

## Model Research
* Converted the Darknet based YOLOv3-tiny model using [Mystic's method](https://github.com/mystic123/tensorflow-yolo-v3.git)
```
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
```
pradan@pradan-HP-15-Notebook-PC:/opt/intel/openvino_2020.1.023/deployment_tools/model_optimizer$ python3 mo_tf.py --input_model ~/yolo-models/mystic_frozen_darknet_yolov3_tiny_model.pb --tensorflow_use_custom_operations_config extensions/front/tf/yolo_v3_tiny.json --batch 1 --output_dir ~/Desktop

```
The conversion was successfully executed, as shown in the log:
```
[ SUCCESS ] Generated IR version 10 model.
[ SUCCESS ] XML file: /home/pradan/Desktop/mystic_frozen_darknet_yolov3_tiny_model.xml
[ SUCCESS ] BIN file: /home/pradan/Desktop/mystic_frozen_darknet_yolov3_tiny_model.bin
[ SUCCESS ] Total execution time: 58.47 seconds.
[ SUCCESS ] Memory consumed: 516 MB.

```

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model fo
### References
* [OpenVINO Toolkit API Classes](https://docs.openvinotoolkit.org/2019_R3/ie_python_api.html)
