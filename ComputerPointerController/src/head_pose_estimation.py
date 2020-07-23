#2
from openvino.inference_engine import IENetwork, IECore, IEPlugin
from time import time
import logging as log
import cv2
import numpy as np

log.basicConfig(format='[INFO] \t %(message)s', level=log.INFO)

class head_pose_estimation:
    '''
    Class for the head_pose_estimation.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_name = model_name
        self.model_weights = model_name+'.bin'
        self.model_structure = model_name+'.xml'

    def load_model(self):#1
        load_time = time()
        self.model = IENetwork(self.model_structure, self.model_weights)
        ie = IECore()
        plugin = IEPlugin(device="CPU")
        self.exec_net = plugin.load(self.model)
        self.net = ie.load_network(network=self.model, device_name='CPU', num_requests=1)
        return time()-load_time

    def predict(self, image):#4
        input_dict = {self.input_blob:image}
        infer_time = time()
        self.exec_net.start_async(request_id=0, inputs=input_dict)
        return time()-infer_time
        #log.info("Inference Complete in {:.4f} s".format(time()-infer_time))

    def check_model(self):#2
        self.input_blob = next(iter(self.model.inputs))
        self.output_blob = next(iter(self.model.outputs))

    def preprocess_input(self, frame):#3
        """
        [1x3x60x60] - An input image in [1xCxHxW] format. Expected color order is BGR.
        """
        b, c, h, w = self.model.inputs[self.input_blob].shape
        log.debug("\nModel: HEAD POSE ESTIMATION")
        log.debug(f"Input Blob Shape: {self.model.inputs[self.input_blob].shape}")
        pframe = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # (H,W,c-BGR)
        pframe = cv2.resize(pframe, (w,h)) # (h,w,c-BGR)
        pframe = pframe.transpose((2,0,1)) #(c-BGR,h,w)
        pframe = pframe.reshape((b,c,h,w)) #(b,c-BGR,h,w)
        return  pframe

    def wait(self):
        ### Wait for the request to be complete. ###
        status = self.exec_net.requests[0].wait(-1)
        return status

    def preprocess_output(self):#6
        '''
        Output layer names in Inference Engine format:

        name: "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees).
        name: "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees).
        name: "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees).
        '''

        output = self.exec_net.requests[0].outputs
        for layer_name, out_blob in output.items():
            log.debug("Layer:{}\tOutput Blob Shape:{}".format(layer_name, out_blob.shape))
        return output["angle_y_fc"], output["angle_p_fc"], output["angle_r_fc"]
