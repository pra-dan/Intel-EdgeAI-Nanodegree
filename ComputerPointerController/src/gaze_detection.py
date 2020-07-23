#4
from openvino.inference_engine import IENetwork, IECore, IEPlugin
from time import time
import logging as log
import cv2
import numpy as np

log.basicConfig(format='[INFO] \t %(message)s', level=log.INFO)

class gaze_detection:
    '''
    Class for the gaze_detection.
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

    def predict(self, input_dict):#4
        infer_time = time()
        self.exec_net.start_async(request_id=0, inputs=input_dict)
        return time()-infer_time
        #log.info("Inference Complete in {:.4f} s".format(time()-infer_time))

    def check_model(self):#2
        self.input_blob = next(iter(self.model.inputs))
        self.output_blob = next(iter(self.model.outputs))

    def preprocess_input(self, face, left_eye_coords, right_eye_coords, head_pose_angles):#3
        """
        Blob left_eye_image and the shape [1x3x60x60] in the format [BxCxHxW]
        Blob right_eye_image and the shape [1x3x60x60] in the format [BxCxHxW]
        Blob head_pose_angles and the shape [1x3] in the format [BxC]
        """
        input_dict = {}
        #print(f"input_blobs: {self.model.inputs.items()}")
        log.debug("\nModel: GAZE DETECTION")
        for layer_name, input_blob in self.model.inputs.items():
            frame = face
            log.debug(f"Input Blob {layer_name}\t Shape: {input_blob.shape}")
            if (layer_name == 'head_pose_angles'):
                input_dict[layer_name] = np.array(head_pose_angles).reshape((input_blob.shape))
                continue # Processed already

            b, c, h, w = input_blob.shape
            # Cropping Eye Section
            #print(f"old frame shape {frame.shape}\n left coords {left_eye_coords} \nrt_coords {right_eye_coords}")
            if(layer_name == 'left_eye_image'):
                xmin = left_eye_coords[0]; xmax = left_eye_coords[2]
                ymin = left_eye_coords[3]; ymax = left_eye_coords[1]
                frame = frame[ymin:ymax , xmin:xmax]
                #print(f"new left_eye_box shape {frame.shape}")
            if(layer_name == 'right_eye_image'):
                xmin = right_eye_coords[0]; xmax = right_eye_coords[2]
                ymin = right_eye_coords[3]; ymax = right_eye_coords[1]
                frame = frame[ymin:ymax , xmin:xmax]
                #print(f"new right_eye_image shape {frame.shape}")

            pframe = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # (H,W,c-BGR)
            pframe = cv2.resize(pframe, (w,h)) # (h,w,c-BGR)
            pframe = pframe.transpose((2,0,1)) #(c-BGR,h,w)
            pframe = pframe.reshape((b,c,h,w)) #(b,c-BGR,h,w)
            input_dict[layer_name] = pframe
        return  input_dict

    def wait(self):
        ### Wait for the request to be complete. ###
        status = self.exec_net.requests[0].wait(-1)
        return status

    def preprocess_output(self, face, left_eye_center, right_eye_center):#6
        '''
        The net outputs a blob "gaze_vector" with the shape: [1, 3], containing
        Cartesian coordinates of gaze direction vector.
        The output vector is not normalized and has non-unit length.
        '''
        #orig_h, orig_w = face.shape[:-1]
        #print(f"face.shape {face.shape}")
        #boxed_faces = []

        output = self.exec_net.requests[0].outputs
        for layer_name, out_blob in output.items():
            log.debug("Layer:{}\tOutput Blob Shape:{}".format(layer_name, out_blob.shape))
            x, y, z = out_blob[0]
            #print(f"vector: {out_blob[0]}")
            """
            Syntax:
            cv2.arrowedLine(image, start_point, end_point, color[, thickness[, line_type[, shift[, tipLength]]]])

            Refer to https://stackoverflow.com/questions/32138637/why-does-positive-y-axis-goes-down-positive-x-axis-go-right
            """
            cv2.arrowedLine(face, (left_eye_center[0],left_eye_center[1]),
                            (left_eye_center[0] + int(x*100),
                            left_eye_center[1] + int(-y*100)), (0, 0, 255), 2, tipLength = 0.5)
            cv2.arrowedLine(face, (right_eye_center[0],right_eye_center[1]),
                            (right_eye_center[0] + int(x*100),
                            right_eye_center[1] + int(-y*100)), (0, 0, 255), 2, tipLength = 0.5)
        cv2.imwrite("landmarks_result.png", face)
        try:
            return face, x,y,z
        except:
            log.error(f"!!! The Model {self.model_name} could not find any Gazing Person")
