"""
This class will take control of the whole process of training or testing Segmentation models
"""

import tensorflow as tf

from models import *
from train import *
# import train_ros
from test import *
from utils.misc import timeit

import os
import pdb
import pickle
from utils.misc import calculate_flops

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Agent:
    """
    Agent will run the program
    Choose the type of operation
    Create a model
    reset the graph and create a session
    Create a trainer or tester
    Then run it and handle it
    """

    def __init__(self, args):
        self.args = args
        self.mode = args.mode

        # Get the class from globals by selecting it by arguments
        self.model = globals()[args.model]
        self.operator = globals()[args.operator]

        self.sess = None

    @timeit
    def build_model(self):

        print('Building Test Network')
        with tf.variable_scope('network') as scope:
            self.train_model = None
            self.model = self.model(self.args)
            self.model.build()
            calculate_flops()

    @timeit
    def run(self):
        """
        Initiate the Graph, sess, model, operator
        :return:
        """
        print("Agent is running now...\n\n")

        # Reset the graph
        tf.reset_default_graph()

        # Create the sess
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

        # Create Model class and build it
        with self.sess.as_default():
            self.build_model()

        # Create the operator
        self.operator = self.operator(self.args, self.sess, self.model, self.model)

        # Run the network
        self.test()

        self.sess.close()
        print("\nAgent is exited...\n")

    def test(self, pkl=False):
        try:
            self.operator.test(pkl)
        except KeyboardInterrupt:
            pass

    def inference(self):
        try:
            self.operator.test_inference()
        except KeyboardInterrupt:
            pass

