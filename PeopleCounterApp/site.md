We start with the pre-processed frame (`pframe`) fed to the inference engine (`ie`). I use the object `infer_network` of the class `Network`. The pre-processing was done as:
```
### Pre-process the image as needed ###
     b, c, h, w = infer_network.get_input_shape()
     pframe = cv2.resize(frame,(w,h))
     pframe = pframe.transpose((2,0,1))
     pframe = pframe.reshape((b,c,h,w))
```
Let's jump directly into the **raw** output of the inference.
The output is obtained as `output = self.exec_net.requests[0].outputs`. This is a dictionary with 2x{Layer, feature_map_values}.
```
for layer_name, out_blob in output.items():
    print(out_blob.shape)
    print("Layer:{}\nOutBlob:{}".format(layer_name, out_blob))

#Layer                                              |  Feature map shape
#detector/yolo-v3-tiny/Conv_12/BiasAdd/YoloRegion   |   (1, 255, 26, 26)
#detector/yolo-v3-tiny/Conv_9/BiasAdd/YoloRegion    |   (1, 255, 13, 13)
```
> Originally, YOLOv3 model includes feature extractor called Darknet-53 with three branches (for v3 and 2 branches for v3-tiny) at the end that make detections at three different scales. These branches must end with the YOLO Region layer. (named as simply YOLO)
Region layer was first introduced in the DarkNet framework. Other frameworks, including TensorFlow, do not have the Region implemented as a single layer, so every author of public YOLOv3 model creates it using simple layers. This badly affects performance. For this reason, the main idea of YOLOv3 model conversion to IR is to cut off these custom Region-like parts of the model and complete the model with the Region layers where required. (Source)[https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html]

Before Converting to IR using OpenVINO.

![https://github.com/PrashantDandriyal/Intel-EdgeAI-Nanodegree/blob/b_done_finding_stats/PeopleCounterApp/resources/yolo_actual.gif](yolo_actual)

Now, it seems lucid why we obtained these two layers as output from the Inference Engine. Pre-conversion to IR, they are named as simply _YOLO_ layers while post-conversion, they are named as _YoloRegion_.

* Now, we go on using these 2 layers by first finding their parameters. The _yolov3-tiny.cfg_ is the source of all these parameters. We just need to pick them from this file manually OR use the `.xml` and `.bin`. We have already initialised the net as:
```
# Read the IR as a IENetwork
self.net = IENetwork(model = model_xml, weights = model_bin)
```
These params are extracted from this net as `self.net.layers[layer_name].params`. The toggle switch between these 2 methods is defined in the class definition:
```
class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolov3-tiny.cfg file (Look in the project folder). If the params can't be extracted automatically, use these hard-coded values.
    def __init__(self, param, side):
        self.num = 3 if 'num' not in param else int(param['num'])
        self.coords = 4 if 'coords' not in param else int(param['coords'])
        self.classes = 80 if 'classes' not in param else int(param['classes'])
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]

        if 'mask' in param:
            mask = [int(idx) for idx in param['mask'].split(',')]
            self.num = len(mask)

            # Collect pairs of anchors to mask/use
            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors

        self.side = side    # 26 for first layer and 13 for second
        self.isYoloV3 = 'mask' in param  # Weak way to determine but the only one.

        def log_params(self):
            params_to_print = {'classes': self.classes, 'num': self.num, 'coords': self.coords, 'anchors': self.anchors}
            [log.info("         {:8}: {}".format(param_name, param)) for param_name, param in params_to_print.items()]

```
To understand the _mask_ mentioned here, actually, the `.cfg` file provides 6 pairs of anchors. These anchors are divided among these 2 feature (output) layers in a pre-determined fashion; the parameter `param` stores this info. To look into the `param` attribute of both these feature layers:
```
# Layer 1
Params: {'anchors': '10,14,23,27,37,58,81,82,135,169,344,319', 'axis': '1', 'classes': '80', 'coords': '4', 'do_softmax': '0', 'end_axis': '3', 'mask': '0,1,2', 'num': '6'}

# Layer 2
Params: {'anchors': '10,14,23,27,37,58,81,82,135,169,344,319', 'axis': '1', 'classes': '80', 'coords': '4', 'do_softmax': '0', 'end_axis': '3', 'mask': '3,4,5', 'num': '6'}
```
The attribute `mask` helps in distributing/allocating the anchors between the layers. Post-process params or the objects of the class `YoloParams`look like:
```
# Layer 1
[ INFO ] Layer detector/yolo-v3-tiny/Conv_12/BiasAdd/YoloRegion parameters:
[ INFO ]          classes : 80
[ INFO ]          num     : 3
[ INFO ]          coords  : 4
[ INFO ]          anchors : [10.0, 14.0, 23.0, 27.0, 37.0, 58.0]

# Layer 2
[ INFO ] Layer detector/yolo-v3-tiny/Conv_9/BiasAdd/YoloRegion parameters:
[ INFO ]          classes : 80
[ INFO ]          num     : 3
[ INFO ]          coords  : 4
[ INFO ]          anchors : [81.0, 82.0, 135.0, 169.0, 344.0, 319.0]
```
This log is dumped by `log_params` in the above class. Another important element in the class definition is `self.isYoloV3 = 'mask' in param`. This simply helps us to determine whether the model being used is v3 or not. Actually, the `mask` is exclusive to YOLOv3 and tiny version. Previous versions lack it.  

... Understanding the excerpt:
* Getting Bounding box coordinates
We loop through the 26x26 and 13x13 values just like we would have done using 2 nested loops. But this is just another method.
```
for i in range(side_square):
      row = i // params.side
      col = i % params.side
      # More code...
```
The working is visualised as:
![https://github.com/PrashantDandriyal/Intel-EdgeAI-Nanodegree/blob/b_done_finding_stats/PeopleCounterApp/resources/avoidingNestedLoop.gif](avoidingNestedLoop)
It is done for both the layers.
