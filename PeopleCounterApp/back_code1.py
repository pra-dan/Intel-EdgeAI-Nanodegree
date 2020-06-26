"""People Counter."""
# Run:
# python3 main.py -m popo_models/yolo-v2-tiny-coco.xml -i images/sample_dog.jpg -d CPU -pt 0.5 -cl coco.names
import os
import sys
import time
import socket
import json
import cv2
import numpy as np
from math import exp

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
    return parser

# Parse the classes file into a list
def class_file_parser(args):
    labels_list = open(args.class_file).read().strip().split("\n")
    # Assign unique colour to each class
    colors = np.random.randint(0, 255, size=(len(labels_list), 3), dtype="uint8")
    return labels_list, colors

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
            print("bo ");print(bot_x, bot_y)

            cv2.rectangle(image,
                        (top_x, top_y),
                        (bot_x, bot_y),
                        fontColor)

            cv2.putText(image,
                      box_cls,
                      (top_x, top_y-5),
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
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Get classes_list
    class_list, colors = class_file_parser(args)
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.cpu_extension)

    ### Handle the input stream ###
    infer_network.get_input_shape()
    # Grab and open video capture
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    ### Loop until stream is over ###
    while cap.isOpened():
        ### Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break

        key_pressed = cv2.waitKey(60)

        ### Pre-process the image as needed ###
        # YOLO v2-tiny accept 416x416. Let's ask the input_blob for this info
        # cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

        b, c, h, w = infer_network.get_input_shape()
        scale = 0.00392
        blob = cv2.dnn.blobFromImage(frame, scale, (416,416), (0,0,0), True, crop=False)

        frame_copy = cv2.resize(frame, (w,h))
        frame = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)

        frame = frame.transpose((2,0,1))
        frame = frame.reshape((b,c,h,w))
        np_img = np.copy(frame)
        np_img = np_img.astype('float32')
        np_img /= 255.0
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        np_img = np.copy(frame) # numpy copy of the frame
        np_img = cv2.resize(np_img, (w,h))
        # [N,H,W,C] for TensorFlow* models
        np_img = np_img.transpose((2,0,1))  #(Gives(C, H, W)
        #Add another dimension to it
        np_img = np_img.astype('float32')
        np_img /= 255.0
        np_img = np_img.reshape(1, 3, h, w)
        """
        ### Start asynchronous inference for specified request ###
        infer_network.async_inference(blob)

        ### Wait for the result ###
        if(infer_network.wait() == 0):
            ### Get the results of the inference request ###
            '''
            The output of YOLOv2-tiny is 13x13x425 (takes in 416x416 as input;
            gives 13x13 output)
            > the CNN reduces the original image to a 13x13x425, where each of
             the 13x13 is a prediction to a grid in the original image.
            The v2-tiny has 80 classes and 5 bounding box coordinates,
            so, (5 anchors *(80 classes + 5)) = 425 | (B * (Classes + 5)) = D or depth
            Read more here: https://stackoverflow.com/a/50570204/9625777
            '''
            result = infer_network.get_output() # Numpy array
            result = result.reshape(1,13,13,425)
            print(result.shape)

            ### TODO: Extract any desired stats from the results ###
            #for cl in range(5,84):
            #    print(result[0][0][0][cl])
            # Refer to this diagram used for understaning below process:
            #https://miro.medium.com/max/1400/1*EROcHT5vKtRQqk0Vfx0Gdw.png
            big_list = []
            for i in range(result.shape[0]):    # 1
                for j in range(result.shape[1]):    # 13
                    for k in range(result.shape[2]):    # # 13
                        for l in range(0, result.shape[3], Anchors+Classes):    # Gives beg indices
                            prob = result[i][j][k][l+4]; #print("prob"+str(prob))
                            if(prob >= prob_threshold):
                                t_x = result[i][j][k][l]; #print("x"+str(x))
                                t_y = result[i][j][k][l+1]; #print("y"+str(y))
                                t_w = result[i][j][k][l+2]; #print("w"+str(w))
                                t_h = result[i][j][k][l+3]; #print("h"+str(h))

                                tx, ty, tw, th, tc = result[i, j, k, l, :5]
                                # Now the remaining terms (l+5:l+85) are 80 Classes
                                index_class = np.argmax(result[i][j][k][l+5:l+85], axis=0)

                                print(str(class_list[index_class]))
                                #print("l: "+str(l)+"  Class: "+str(index_class)+"   ->  "+class_list[index_class]) #-(l+5)
                                anchor_index = int(l//(Anchors+Classes))  # 0 1 2 3 4
                                print(str(l)+", "+str(anchor_index))

                                # Calculating X and Y coordinates
                                stride = 416 // 13    # = 32 | The output feature map is : 13x13x425
                                center_x = int(t_x * w)   #
                                center_y = int(t_y * h)   # C
                                w = int(t_w * w)
                                h = int(t_h * h)
                                #box_top_x = (t_x + offset_x)*stride
                                #box_top_y = (t_y + offset_y)*stride

                                x = center_x - w / 2
                                y = center_y - h / 2
                                # Calcuating W and H for the box
                                #box_w = anchor_sets[anchor_index] * exp(t_w) * stride
                                #box_h = anchor_sets[anchor_index+1] * exp(t_h) * stride

                                big_list.append([class_list[index_class],x,y,w,h])
                                print([str(prob), class_list[index_class],x,y,w,h])
            # Draw boxes over image and save
            draw_box(frame_copy, big_list, colors)


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
    start = time.perf_counter()
    infer_on_stream(args, client)
    time_took = time.perf_counter() - start
    print(f"Time took: {time_took:.6f}s")

if __name__ == '__main__':
    main()

