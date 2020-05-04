"""People Counter."""
# Run:
#  python3 main.py -m popo_models/mystic_frozen_darknet_yolov3_tiny_model.xml -i /opt/intel/openvino_2020.1.023/Intel-EdgeAI-Nanodegree/PeopleCounterApp/images/sample_dog.jpg -cl coco.names
import os
import sys
from time import time
import socket
import json
import cv2
import numpy as np
from math import exp

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

import logging
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

Anchors = 5
anchor_sets =  [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
# Link: https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-tiny.cfg
Classes = 80


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-cl", "--class_file", type=str, required=True,
                        help="Specify the file having Class names")
    parser.add_argument("-iout", "--iou_threshold", help="Optional. Intersection over union threshold for overlapping "
                                                       "detections filtering", default=0.4, type=float)
    return parser

# Parse the classes file into a list
def class_file_parser(args):
    labels_map = open(args.class_file).read().strip().split("\n")
    # Assign unique colour to each class
    colors = np.random.randint(0, 255, size=(len(labels_map), 3), dtype="uint8")
    return labels_map, colors

def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union

def non_maximal_suppression(thresholded_predictions,iou_threshold):

    nms_predictions = []
    nms_predictions.append(thresholded_predictions[0])
    # thresholded_predictions[0] = [x1,y1,x2,y2]

    i = 1
    while i < len(thresholded_predictions):
        n_boxes_to_check = len(nms_predictions)
        #print('N boxes to check = {}'.format(n_boxes_to_check))
        to_delete = False

        j = 0
        while j < n_boxes_to_check:
            curr_iou = iou(thresholded_predictions[i][0],nms_predictions[j][0])
            if(curr_iou > iou_threshold ):
                to_delete = True
                #print('Checking box {} vs {}: IOU = {} , To delete = {}'.format(thresholded_predictions[i][0],nms_predictions[j][0],curr_iou,to_delete))
                j = j+1

                if to_delete == False:
                    nms_predictions.append(thresholded_predictions[i])
                    i = i+1

                    return nms_predictions

def draw_box(image, big_list, colors):
    #image = cv2.imread(img)
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    fontColor              = (0,255,0)
    lineType               = 2
    thickness = 3
    #print(image.shape)  #(3, 416, 416)
    _, img_h, img_w = image.shape
    #color = [c colors
    for i in big_list:
        box_cls = i[0]; print(box_cls)
        box_x = i[1]; print(box_x)
        box_y = i[2]; print(box_y)
        box_w = i[3]; print(box_w)
        box_h = i[4]; print(box_h)



        if(box_x >=0 and box_y >= 0 and box_w>=0 and box_h>=0 and box_cls == "dog"):
            top_x = int(box_x)
            top_y = int(box_y)
            bot_y = int(top_y+box_h)
            bot_x = int(top_x+box_w);
            print(bot_x, bot_y)
        print("bo ");print(i[1], i[2], i[3], i[4])
        cv2.rectangle(image,
                    (int(i[1]), int(i[2])),
                    (int(i[3]), int(i[4])),
                    fontColor)

        cv2.putText(image,
                  box_cls,
                  (i[1], i[2]-5),
                  font,
                  fontScale,
                  fontColor,
                  lineType)

    # Saving the image
    cv2.imwrite("output.png",image)
    cv2.imshow("output", image)

def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = None

    return client


def infer_on_stream(args, client):
    # Get classes_list
    labels_map, colors = class_file_parser(args)

    # Initialise the class
    infer_network = Network()
    prob_threshold = args.prob_threshold    # Set Probability threshold for detections

    ### Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.cpu_extension)

    ### Handle the input stream ###
    # Grab and open video capture
    infer_network.get_input_shape()
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    ### Loop until stream is over ###
    while cap.isOpened():
        ### Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break

        key_pressed = cv2.waitKey(60)

        # Prepare for inference
        start_time = time()
        ### Pre-process the image as needed ###
        b, c, h, w = infer_network.get_input_shape()
        pframe = cv2.resize(frame,(w,h))
        pframe = pframe.transpose((2,0,1))
        pframe = pframe.reshape((b,c,h,w))

        ### Start asynchronous inference for specified request ###
        infer_time = infer_network.async_inference(pframe)
        log.info("Inference Complete in {}".format(infer_time))
        ### Wait for the result ###
        if(infer_network.wait() == 0):
            ### Get the results of the inference request ###
            '''
            The output of YOLOv2-tiny is 255x13x13 (takes in 416x416 as input;
            gives 13x13 output), the 13x13 is a prediction to a grid in the resized image.
            The v2-tiny has 80 classes and 5 bounding box coordinates,
            so, (3 anchors *(80 classes + 5)) = 255 | (B * (Classes + 5)) = D or depth
            Read more here: https://stackoverflow.com/a/50570204/9625777
            '''
            # Get output blobs/objects
            objects = infer_network.get_output(args.prob_threshold,frame, pframe) # A dict with 2 layers

            # Filtering overlapping boxes with respect to the --iou_threshold CLI parameter
            objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
            for i in range(len(objects)):
                if objects[i]['confidence'] == 0:
                    continue
                for j in range(i + 1, len(objects)):
                    if intersection_over_union(objects[i], objects[j]) > args.iou_threshold:
                        objects[j]['confidence'] = 0

            # Drawing objects with respect to the --prob_threshold CLI parameter
            objects = [obj for obj in objects if obj['confidence'] >= args.prob_threshold]

            origin_im_size = frame.shape[:-1]
            for obj in objects:
                # Validation bbox of detected object
                if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
                    continue
                color = (int(min(obj['class_id'] * 12.5, 255)),
                         min(obj['class_id'] * 7, 255), min(obj['class_id'] * 5, 255))
                det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else \
                    str(obj['class_id'])
                """
                if args.raw_output_message:
                    log.info(
                        "{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} | {} ".format(det_label, obj['confidence'], obj['xmin'],
                                                                                  obj['ymin'], obj['xmax'], obj['ymax'],
                                                                                  color))
                """
                cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), (255,255,255), 2)
                cv2.putText(frame,
                            "#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',
                            (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            cv2.imshow("DetectionResults", frame)
            cv2.imwrite("DetectionResults.png", frame)
            exit()

            ### TODO: Extract any desired stats from the results ###
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

        ### TODO: Send the frame to the FFMPEG server ###

        ### TODO: Write an output image if `single_image_mode` ###


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    fin_time = time()
    infer_on_stream(args, client)
    time_took = time() - fin_time
    log.info(f"Total execution Time: {time_took:.6f}s")
    print(time_took)

if __name__ == '__main__':
    main()
