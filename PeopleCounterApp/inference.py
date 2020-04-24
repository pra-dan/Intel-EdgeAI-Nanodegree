#!/usr/bin/env python3

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore, IEPlugin

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
        model_bin = os.path.splitext(model_xml[0] + ".bin")

        # Initialise the plugin
        self.plugin = IECore()
        # Read the IR as a IENetwork
        self.net = IENetwork(model = model_xml, weights = model_bin)
        # This network is actually our model and all relevant info of
        # the model can be extracted from it.

        ### Check for supported layers ###
        print("\nThe supported layers in the model are: ")
        supp_layers = plugin.get_supported_layers(net)
        print(str(supp_layers))
        unsupp_layers = [u for u in net.layers.keys() if u not in supp_layers]
        if len(unsupp_layers) != 0:
            print("\nFound Unsupported layers: {}".format(string(unsupp_layers)))

        ### Add any necessary extensions for running these unsupported layer###
        if(CPU_EXTENSION and "CPU" in device):
            self.plugin.add_extension(extension_path = CPU_EXTENSION, device)

        ### Return the loaded inference plugin ###
        self.exec_net = self.plugin.load_network(self.net, device)
        print("\nIR successfully loaded into Inference Engine.")

        self.input_blob = next(iter(self.net.inputs))
        self.output_blob = next(iter(self.net.outputs))

        ### Note: You may need to update the function parameters. ###
        return

    def get_input_shape(self):
        ### Return the shape of the input layer ###
        print("\nThe input blob shape was found as: {}".format(string(self.net.inputs[self.input_blob].shape)))
        return

    def async_inference(self, img):
        ### Start an asynchronous request ###
        self.exec_net.start_async(request_id=0, inputs={self.input_blob: img})
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return

    def wait(self):
        ### Wait for the request to be complete. ###
        status = self.exec_net.requests[0].wait(-1)

        ### Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return status

    def get_output(self):
        ### Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.exec_net.output_blob
