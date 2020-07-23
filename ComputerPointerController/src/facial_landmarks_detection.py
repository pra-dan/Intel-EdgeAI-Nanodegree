#3
from openvino.inference_engine import IENetwork, IECore, IEPlugin
from time import time
import logging as log
import cv2
import numpy as np

log.basicConfig(format='[INFO] \t %(message)s', level=log.INFO)

class facial_landmarks_detection:
    '''
    Class for the facial_landmarks_detection.
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

    def check_model(self):#2
        self.input_blob = next(iter(self.model.inputs))
        self.output_blob = next(iter(self.model.outputs))

    def preprocess_input(self, frame):#3
        """
        Name: "data" , shape: [1x3x48x48]
        An input image in the format [BxCxHxW] , Channels in BGR format
        """
        b, c, h, w = self.model.inputs[self.input_blob].shape
        log.debug("\nModel: FACIAL LANDMARKS DETECTION")
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

    def preprocess_output(self, face):#6
        '''
        The net outputs a blob with the shape: [1, 10, 1, 1],
        containing a row-vector of 10 floating point values for five landmarks
        coordinates in the form (x0, y0, x1, y1, ..., x5, y5).
        All the coordinates are normalized to be in range [0,1].
        '''
        orig_h, orig_w = face.shape[:-1]
        #print(f"face.shape {face.shape}")

        output = self.exec_net.requests[0].outputs
        for layer_name, out_blob in output.items():
            log.debug("Layer:{}\tOutput Blob Shape:{}".format(layer_name, out_blob.shape))
        flat_blob = out_blob.reshape(10)
        #print(f"shape {out_blob.shape} \n flat blob :{flat_blob}")
        five_coord_pairs = [[orig_w*flat_blob[i], orig_h*flat_blob[i+1]] for i in range(0, out_blob.shape[1], 2)]

        # Drawing points
        shift = int(orig_h/14)
        for idx, pt in enumerate(five_coord_pairs):
            cx = int(pt[0])
            cy = int(pt[1])
            cv2.circle(face, (cx,cy), 2, (255,255,255), 3)

            # Only for eyes
            if(idx == 0 or idx == 1):
                xmin = cx - shift
                xmax = cx + shift
                ymin = cy + shift
                ymax = cy - shift

                if(idx ==0 ): # Left Eye
                    left_eye_center = cx,cy
                    left_eye_box = [xmin,ymin,xmax,ymax]
                else:
                    right_eye_center = cx, cy
                    right_eye_box = [xmin,ymin,xmax,ymax]
            cv2.rectangle(face,
                        (xmin, ymin), (xmax, ymax),
                        (200,0,0),3)
        cv2.imwrite("landmarks_result.png", face)
        try:
            return left_eye_box, right_eye_box, face, left_eye_center, right_eye_center
        except:
            log.error(f"!!! The Model {self.model_name} could not find any Landmarks")
