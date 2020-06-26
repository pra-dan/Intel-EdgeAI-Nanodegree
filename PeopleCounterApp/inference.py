import os
import sys
import logging as log
from time import time
from math import exp
import numpy as np

import logging
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()
from openvino.inference_engine import IENetwork, IECore, IEPlugin


class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolov3-tiny.cfg file (Look in the project folder)
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

            # Collect pairs of anchors to mask
            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors

        self.side = side    # 26 for first layer and 13 for second
        self.isYoloV3 = 'mask' in param  # Weak way to determine but the only one.


    def log_params(self):
        params_to_print = {'classes': self.classes, 'num': self.num, 'coords': self.coords, 'anchors': self.anchors}
        [log.info("         {:8}: {}".format(param_name, param)) for param_name, param in params_to_print.items()]


def entry_index(side, coord, classes, location, entry):
    side_power_2 = side ** 2
    n = location // side_power_2
    loc = location % side_power_2
    return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)


def scale_bbox(x, y, h, w, class_id, confidence, h_scale, w_scale):
    """scale = np.array([min(w_scale/h_scale, 1), min(h_scale/w_scale, 1)])
    offset = 0.5*(np.ones(2) - scale)
    x, y = (np.array([x, y]) - offset) / scale
    width, height = np.array([w, h]) / scale"""
    #print(f"x{x}, y{y}, w{w}, h{h}")
    xmin = int((x - w / 2) * w_scale)
    ymin = int((y - h / 2) * h_scale)
    xmax = int(xmin + w * w_scale)
    ymax = int(ymin + h * h_scale)

    print(f"x{xmin}, y{ymin}, xm{xmax}, ym{ymax}")
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)


def parse_yolo_region(blob, resized_image_shape, original_im_shape, params, threshold,labels_map):
    # ------------------------------------------ Validating output parameters ------------------------------------------
    _, _, out_blob_h, out_blob_w = blob.shape   # [26, 26] and [13, 13]
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    #print(f"predictions shape{blob.shape}")
    orig_im_h, orig_im_w = original_im_shape    # 416
    resized_image_h, resized_image_w = resized_image_shape  # 416
    objects = list()

    size_normalizer = (resized_image_w, resized_image_h) if params.isYoloV3 else (params.side, params.side)

    for oth in range(0, blob.shape[1], 85):    # 255
        for row in range(blob.shape[2]):       # 13
            for col in range(blob.shape[3]):   # 13
                #print(f"l {l}")
                info_per_anchor = blob[0, oth:oth+85, row, col] #print("prob"+str(prob))
                x, y, width, height, prob = info_per_anchor[:5]

                if(prob < threshold):
                    continue

                # Now the remaining terms (l+5:l+85) are 80 Classes
                class_id = np.argmax(info_per_anchor[5:])

                print(f"class @ 0.1 : {labels_map[class_id]} \t prob:{prob}")
                #print("l: "+str(l)+"  Class: "+str(index_class)+"   ->  "+class_list[index_class]) #-(l+5)
                # Process raw value
                """
                try exchangin col and row
                """
                x = (col + x) / params.side
                y = (row + y) / params.side
                # Value for exp is very big number in some cases so following construction is using here
                try:
                    width = exp(width)
                    height = exp(height)
                except OverflowError:
                    continue
                # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
                n = int(oth/85)
                #print(f"n: {n}, norm..{size_normalizer}")
                width = width * params.anchors[2 * n] / size_normalizer[0]
                height = height * params.anchors[2 * n + 1] / size_normalizer[1]

                objects.append(scale_bbox(x=x, y=y, h=height, w=width, class_id=class_id, confidence=info_per_anchor[class_id],
                                          h_scale=orig_im_h, w_scale=orig_im_w))
    #print(f"objects at enpacking: \t{objects}")
    return objects

class Network:
    """
    Load and configure inference plugins for the specified target devices
    and performs synchronous and asynchronous modes for the specified infer requests.
    """
    def __init__(self):
        ### Initialize any class variables desired ###
        # Inference Engine instance
        self.plugin = None

    def load_model(self, model, device="CPU", CPU_EXTENSION=None):
        ### Load the model ###
        model_xml = model
        # print(os.path.splitext(model_xml))
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Note: These 3 things are utmostly important !!!
        # Load the Inference Engine API
        ie = IECore()
        ### Add any necessary extensions for running these unsupported layer###
        if(CPU_EXTENSION):
            self.ie.add_extension(CPU_EXTENSION, "CPU")
        # Initialise the plugin
        plugin = IEPlugin(device="CPU")
        # Read the IR as a IENetwork
        self.net = IENetwork(model = model_xml, weights = model_bin)
        # This network is actually our model and all relevant info of
        # the model can be extracted from it.

        ### Check for supported layers ###
        #print("\nThe supported layers in the model are: ")
        supp_layers = plugin.get_supported_layers(self.net)
        #print(str(supp_layers))
        unsupp_layers = [u for u in self.net.layers.keys() if u not in supp_layers]
        if len(unsupp_layers) != 0:
            print("\nFound Unsupported layers: {}".format(string(unsupp_layers)))

        ### Return the loaded inference plugin ###
        self.exec_net = plugin.load(self.net)
        print("\nIR successfully loaded into Inference Engine.")

        self.input_blob = next(iter(self.net.inputs))
        self.output_blob = next(iter(self.net.outputs))
        return

    def get_input_shape(self):
        ### Return the shape of the input layer ###
        print("Input blob shape: {}".format(str(self.net.inputs[self.input_blob].shape)))
        print("Output blob shape: {}".format(str(self.net.outputs[self.output_blob].shape))) # (13, 13, 425)
        return self.net.inputs[self.input_blob].shape

    def async_inference(self, img):
        ### Start an asynchronous request ###
        infer_time_beg = time()
        self.exec_net.start_async(request_id=0, inputs={self.input_blob: img})
        return time() - infer_time_beg

    def wait(self):
        ### Wait for the request to be complete. ###
        status = self.exec_net.requests[0].wait(-1)
        return status

    def get_output(self, prob, frame, pframe,labels_map):
        ### Extract and return the output results
        output = self.exec_net.requests[0].outputs
        start_time = time()
        objects = list()
        for layer_name, out_blob in output.items():
            # For each of 2 output layers, get their params
            layer_params = YoloParams(self.net.layers[layer_name].params, out_blob.shape[2])
            #print("net Params: {}\t shape[2]{}".format(self.net.layers[layer_name].params, out_blob.shape[2]))
            log.info("Layer {} parameters: ".format(layer_name))
            layer_params.log_params()
            objects += parse_yolo_region(out_blob, pframe.shape[2:],    # [Feature map outputs, 416, 416, YoloParams objects]
                                         frame.shape[:-1], layer_params,
                                         prob, labels_map)
            print(f"In Layer {layer_name}")
            print("Detected Objects")
            for i in objects:
                print(f"{i}")

        parsing_time = time() - start_time

            #print("Layer:{}\nOutBlob:{}".format(layer_name, out_blob))
            #print(out_blob.shape)
        return objects
