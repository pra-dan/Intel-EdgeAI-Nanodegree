#1
from openvino.inference_engine import IENetwork, IECore, IEPlugin
from time import time
import logging as log

class face_detection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_name = model_name
        model_weights = model_name+'.bin'
        model_structure = model_name+'.xml'
        log.info(f"\nModel: {self.model_name}")

    def load_model(self):#1
        start = time()
        self.model = IENetwork(self.model_structure, self.model_weights)
        ie = IECore()
        self.net = ie.load_network(network=self.model, device='CPU', num_requests=1)
        log.info("Model Load Time: ".format(time()-start))

    def predict(self, image):#4
        input_dict = {self.input_blob:image}
        infer_time = time()
        self.net.infer(input_dict)
        log.info("Inference Complete in {}".format(infer_time))

    def check_model(self):#2
        self.input_blob = next(iter(self.model.inputs))
        self.output_blob = next(iter(self.model.outputs))

    def preprocess_input(self, image_path):#3
        pframe = cv2.imread(image_path)
        b, c, h, w = self.model.get_input_shape()
        pframe = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # (H,W,c-BGR)
        pframe = cv2.resize(pframe, (w,h), interpolation=cv2.INTER_AREA) # (h,w,c-BGR)
        pframe = pframe.transpose((2,0,1)) #(c-BGR,h,w)
        pframe = pframe.reshape((b,c,h,w)) #(b,c-BGR,h,w)
        return  pframe

    def preprocess_output(self, outputs):#5
        """
        The net outputs blob with shape: [1, 1, N, 7], where N is the number of
        detected bounding boxes. Each detection has the format [image_id, label,
        conf, x_min, y_min, x_max, y_max]
        """
        _,_,N,values = self.output_blob.shape
        out = self.output_blob.reshape((N,values))
        objects = [out[n,:] for n in range(N)]
        return objects
