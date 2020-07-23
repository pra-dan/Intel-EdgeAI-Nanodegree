#1
from openvino.inference_engine import IENetwork, IECore, IEPlugin
from time import time
import logging as log
import cv2
import numpy as np

log.basicConfig(format='[INFO] \t %(message)s', level=log.INFO)

class face_detection:
    '''
    Class for the Face Detection Model.
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
        b, c, h, w = self.model.inputs[self.input_blob].shape
        log.debug(f"\nModel: FACE DETECTION")
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

    def preprocess_output(self,frame,threshold):#6
        '''
        The net outputs blob with shape: [1, 1, N, 7], where N is the number of
        detected bounding boxes. Each detection has the format [image_id, label,
        conf, x_min, y_min, x_max, y_max]
        '''
        orig_h, orig_w = frame.shape[:-1]
        boxed_faces = []

        output = self.exec_net.requests[0].outputs
        for layer_name, out_blob in output.items():
            log.debug("Layer:{}\tOutput Blob Shape:{}".format(layer_name, out_blob.shape))
            _,_,N,values = out_blob.shape
            out = out_blob.reshape((N,values))
            objects = [out[n,:] for n in range(N)]
            for obj in objects:
                if(obj[2] < threshold): continue
                #print(obj)
                xmin = int(obj[3]*orig_w)
                ymin = int(obj[4]*orig_h)
                xmax = int(obj[5]*orig_w)
                ymax = int(obj[6]*orig_h)
                cv2.rectangle(frame,
                            (xmin, ymin), (xmax, ymax),
                            (255,255,255),3)
                boxed_faces.append([xmin, ymin, xmax, ymax])

            cv2.imwrite("res.png",frame)
        return boxed_faces, frame
