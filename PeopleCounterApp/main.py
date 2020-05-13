"""People Counter
# Run:
$ python3 main.py -m popo_models/mystic_frozen_darknet_yolov3_tiny_model.xml -i /opt/intel/openvino_2020.1.023/Intel-EdgeAI-Nanodegree/PeopleCounterApp/resources/pedes_detect.mp4 -pt 0.3 -cl coco.names | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
"""
import os
import sys      # For FFMPEG
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

# uncomment for logging
#import logging
#logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
#log = logging.getLogger()

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 120

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="0: For using webcam, or else, Path to image or video file")
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
    return labels_map

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

def draw_box(frame, labels_map, objects, infer_time):
    origin_im_size = frame.shape[:-1]
    for obj in objects:
        # Validation bbox of detected object
        if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
            continue
        color = (int(min(obj['class_id'] * 12.5, 255)),
                 min(obj['class_id'] * 7, 255), min(obj['class_id'] * 5, 255))
        det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else \
            str(obj['class_id'])

        cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), (255,255,255), 3)
        cv2.putText(frame,
                    str(round(obj['confidence'] * 100, 1)) + ' %',
                    (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    cv2.putText(frame,"Inference Time: "+str(round(infer_time,4)), (2,20), cv2.FONT_HERSHEY_COMPLEX, 1, (125,125,255), 2)

    #cv2.imshow("DetectionResults", frame)
    #cv2.imwrite("DetectionResults.jpg", frame)
    return frame

class stats:
    def __init__(self,fps,client):
        """
        fps: Use FPS to compensate for the time lost when the person is not
            detected during approaching and exit. (corresponds to Frames in a sec)
        cu
        """
        self.fps = fps
        self.client = client
        self.curr_person_count = 0
        self.tota_person_count = 0
        self.duration = self.fps
        self.are_you_counting = False
        self.duration_list = [0]
        self.multiple_checks = self.fps

    def update(self, count):
        self.curr_person_count = count
        # Look for continuously detections (All frames should detect)
        if(self.curr_person_count > 0):
            # Did you detect a person ? If Yes, check that he is detected
            #"multiple_checks"-1 more times
            self.multiple_checks -= 1
            if(self.multiple_checks == 0):
                # Detected the person sufficient times. Now, Reset the check count
                # and consider the detection as a true Person.
                self.multiple_checks = self.fps
                self.duration += 1
                if(self.are_you_counting == False):     # New Person detected
                    self.tota_person_count += 1
                    self.are_you_counting = True
                    # The person has been identified ONCE and never again will I do it again :(
        # Reset the person counter : "multiple_checks" if no person was detected
        # OR if he was not detected continuously for that many times
        elif(self.curr_person_count == 0 and self.are_you_counting == True):
            self.are_you_counting = False
            self.publishResults()
        else:
            self.multiple_checks = 10
            self.duration = 2*self.fps
            self.are_you_counting = False
        """
        Publish this LIVE (unlike the "duration" topic, published only after
        the person leaves). This is done in the accordance with the strange
        way the "webservice/ui/src/features/stats/Stats.jsx" is pre-written.
        That is, the "Total_count" is the number of times the dumps are made to
        the "duration" topic.
        See more here:https://knowledge.udacity.com/questions/130017
        """
        self.client.publish("person", json.dumps({"count": self.curr_person_count, "total": self.tota_person_count}))

    def publishResults(self):
        self.duration_list.append(self.duration)
        #print("Current: {}\t Total: {}\tduration: {}".format(self.curr_person_count, self.tota_person_count, self.duration))
        ### Calculate and send relevant information on ###
        ### current_count, total_count and duration to the MQTT server ###
        ### Topic "person": keys of "count" and "total" ###
        ### Topic "person/duration": key of "duration" ###
        #self.client.publish("person", json.dumps({"count": self.curr_person_count, "total": self.tota_person_count}))
        #print("total: {}\t Dur: {}".format(self.tota_person_count, self.duration))
        self.client.publish("person/duration", json.dumps({"duration": self.duration}))

def connect_mqtt():
    ### Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    # Get classes_list
    labels_map = class_file_parser(args)

    # Initialise the class
    infer_network = Network()
    prob_threshold = args.prob_threshold    # Set Probability threshold for detections

    ### Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.cpu_extension)

    ### Handle the input stream ###
    # Grab and open video capture
    infer_network.get_input_shape()
    cap = cv2.VideoCapture(args.input)

    # Check if webcam as input is supported
    if(cap.isOpened() == False and args.input == 0):
        log.info("The webcam is not supported right now. Please troubleshoot the issue or provide video/image file")
        sys.exit()

    cap.open(args.input)
    # Get the FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    #print(fps)
    # For stats
    stat = stats(fps,client)

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
        log.info(f"Inference Time: {infer_time:.6f}s")
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
            objects = [obj for obj in objects if obj['confidence'] >= args.prob_threshold and labels_map[obj['class_id']] == "person"]
            final_frame = draw_box(frame, labels_map, objects,infer_time)

            ### Extract any desired stats from the results ###
            stat.update(len(objects))
            # The above function also sends the stats to the server

        ### Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(final_frame)
        sys.stdout.flush()
        ### Write an output image if `single_image_mode` ###
        cv2.imwrite("output.jpg",final_frame)

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    cv2.destroyAllWindows()

    # Connect to the MQTT server
    client = connect_mqtt()

    # Perform inference on the input stream
    fin_time = time()
    infer_on_stream(args, client)
    time_took = time() - fin_time
    log.info(f"Total execution Time: {time_took:.6f}s")

    # WrapUp
    client.disconnect()

if __name__ == '__main__':
    main()
