"""People Counter."""
# Run: python3 main.py -m popo_models/yolo-v2-coco.xml -i images/sample_dog.jpg -l $HOME/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libinference_engine_c_api.so -d CPU -pt 0.5  
import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


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
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = None

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.cpu_extension)

    ### Handle the input stream ###
    infer_network.get_input_shape()
    # Grab and open video capture
    cap = cv2.VideoCapture(args.image)
    cap.open(args.image)

    ### Loop until stream is over ###
    while cap.isOpened():
        ### Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        
        key_pressed = cv2.waitKey(60)
        ### Pre-process the image as needed ###
        # YOLO v2-tiny accept 416x416. Let's ask the input_blob for this info
        np_img = np.copy(frame) # numpy copy of the frame
        b, c, h, w = infer_network.get_input_shape()
        np_img = cv2.resize(np_img, (w,h))
        np_img = np_img.transpose((2,0,1))  #(Gives(C, H, W)
        #Add another dimension to it
        np_img = np_img.reshape(1, 3, h, w)

        ### Start asynchronous inference for specified request ###
        infer_network.async_inference(np_img)

        ### Wait for the result ###
        if(infer_network.wait() == 0):
            ### Get the results of the inference request ###
            result = infer_network.get_output() # Numpy array
            print("\nResult"+result)

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
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
