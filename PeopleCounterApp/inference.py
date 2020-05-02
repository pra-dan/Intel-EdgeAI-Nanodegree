#!/usr/bin/env python3

import os
import sys
import logging as log
from time import time
from math import exp
from openvino.inference_engine import IENetwork, IECore, IEPlugin


class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
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

            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors

        self.side = side
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
    xmin = int((x - w / 2) * w_scale)
    ymin = int((y - h / 2) * h_scale)
    xmax = int(xmin + w * w_scale)
    ymax = int(ymin + h * h_scale)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)


def parse_yolo_region(blob, resized_image_shape, original_im_shape, params, threshold):
    # ------------------------------------------ Validating output parameters ------------------------------------------
    _, _, out_blob_h, out_blob_w = blob.shape
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()
    predictions = blob.flatten()
    side_square = params.side * params.side

    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    for i in range(side_square):
        row = i // params.side
        col = i % params.side
        for n in range(params.num):
            obj_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, params.coords)
            scale = predictions[obj_index]
            if scale < threshold:
                continue
            box_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, 0)
            # Network produces location predictions in absolute coordinates of feature maps.
            # Scale it to relative coordinates.
            x = (col + predictions[box_index + 0 * side_square]) / params.side
            y = (row + predictions[box_index + 1 * side_square]) / params.side
            # Value for exp is very big number in some cases so following construction is using here
            try:
                w_exp = exp(predictions[box_index + 2 * side_square])
                h_exp = exp(predictions[box_index + 3 * side_square])
            except OverflowError:
                continue
            # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
            w = w_exp * params.anchors[2 * n] / (resized_image_w if params.isYoloV3 else params.side)
            h = h_exp * params.anchors[2 * n + 1] / (resized_image_h if params.isYoloV3 else params.side)
            for j in range(params.classes):
                class_index = entry_index(params.side, params.coords, params.classes, n * side_square + i,
                                          params.coords + 1 + j)
                confidence = scale * predictions[class_index]
                if confidence < threshold:
                    continue
                objects.append(scale_bbox(x=x, y=y, h=h, w=w, class_id=j, confidence=confidence,
                                          h_scale=orig_im_h, w_scale=orig_im_w))
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
        print("\nThe supported layers in the model are: ")
        supp_layers = plugin.get_supported_layers(self.net)
        print(str(supp_layers))
        unsupp_layers = [u for u in self.net.layers.keys() if u not in supp_layers]
        if len(unsupp_layers) != 0:
            print("\nFound Unsupported layers: {}".format(string(unsupp_layers)))

        ### Return the loaded inference plugin ###
        self.exec_net = plugin.load(self.net)
        print("\nIR successfully loaded into Inference Engine.")

        self.input_blob = next(iter(self.net.inputs))
        self.output_blob = next(iter(self.net.outputs))

        ### Note: You may need to update the function parameters. ###
        return

    def get_input_shape(self):
        ### Return the shape of the input layer ###
        print("\nThe input blob shape was found as: {}".format(str(self.net.inputs[self.input_blob].shape)))
        print("\nThe output blob shape was found as: {}".format(str(self.net.outputs[self.output_blob].shape))) # (13, 13, 425)
        return self.net.inputs[self.input_blob].shape

    def async_inference(self, img):
        ### Start an asynchronous request ###
        infer_time_beg = time()
        self.exec_net.start_async(request_id=0, inputs={self.input_blob: img})
        ### Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return time() - infer_time_beg

    def wait(self):
        ### Wait for the request to be complete. ###
        status = self.exec_net.requests[0].wait(-1)
        return status

    def get_output(self, prob, frame, pframe):
        ### Extract and return the output results
        output = self.exec_net.requests[0].outputs
        start_time = time()
        objects = list()
        for layer_name, out_blob in output.items():
            print(out_blob.shape)
            print("Layer:{}\nOutBlob:{}".format(layer_name, out_blob))
            out_blob = out_blob.reshape(self.net.layers[self.net.layers[layer_name].parents[0]].shape)
            layer_params = YoloParams(self.net.layers[layer_name].params, out_blob.shape[2])
            log.info("Layer {} parameters: ".format(layer_name))
            layer_params.log_params()
            objects += parse_yolo_region(out_blob, pframe.shape[2:],
                                         frame.shape[:-1], layer_params,
                                         prob)
        parsing_time = time() - start_time
            #detector/yolo-v3-tiny/Conv_12/BiasAdd/YoloRegion, and
            #detector/yolo-v3-tiny/Conv_9/BiasAdd/YoloRegion

            #print("Layer:{}\nOutBlob:{}".format(layer_name, out_blob))
            #print(out_blob.shape)
        return objects
