#!/usr/bin/env python3
"""
Trainer class to train Segmentation models
"""

from train.basic_train import BasicTrain

import numpy as np
import tensorflow as tf
import time
from scipy.ndimage import zoom
# import scipy.misc as misc
# import cv2
import rospy
from PIL import Image as Img
# from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import os
import pdb


class Train(BasicTrain):
    """
    Trainer class
    """

    def __init__(self, args, sess, train_model, test_model):
        """
        Call the constructor of the base class
        init summaries
        init loading data
        :param args:
        :param sess:
        :param model:
        :return:
        """
        super().__init__(args, sess, train_model, test_model)
        ##################################################################################
        # Init summaries

        # Summary variables
        self.scalar_summary_tags = ['mean_iou_on_val',
                                    'train-loss-per-epoch', 'val-loss-per-epoch',
                                    'train-acc-per-epoch', 'val-acc-per-epoch']
        self.images_summary_tags = [
            ('train_prediction_sample', [None, self.params.img_height, self.params.img_width * 2, 3]),
            ('val_prediction_sample', [None, self.params.img_height, self.params.img_width * 2, 3])]
        self.summary_tags = []
        self.summary_placeholders = {}
        self.summary_ops = {}
        # init summaries and it's operators
        self.init_summaries()
        # Create summary writer
        self.summary_writer = tf.summary.FileWriter(self.args.summary_dir, self.sess.graph)
        ##################################################################################
        # Init load data and generator
        self.generator = None

        self.test_data = None
        self.test_data_len = None
        self.num_iterations_testing_per_epoch = None
        # self.load_val_data()
        # self.generator = self.test_generator
        self.img = Image()
        # self.bridge = CvBridge()

        self.seg_pub = rospy.Publisher("segmentation", Image, queue_size=1)

        rospy.init_node('whatever')

    def init_summaries(self):
        """
        Create the summary part of the graph
        :return:
        """
        with tf.variable_scope('train-summary-per-epoch'):
            for tag in self.scalar_summary_tags:
                self.summary_tags += tag
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
            for tag, shape in self.images_summary_tags:
                self.summary_tags += tag
                self.summary_placeholders[tag] = tf.placeholder('float32', shape, name=tag)
                self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag], max_outputs=10)

    def add_summary(self, step, summaries_dict=None, summaries_merged=None):
        """
        Add the summaries to tensorboard
        :param step:
        :param summaries_dict:
        :param summaries_merged:
        :return:
        """
        if summaries_dict is not None:
            summary_list = self.sess.run([self.summary_ops[tag] for tag in summaries_dict.keys()],
                                         {self.summary_placeholders[tag]: value for tag, value in
                                          summaries_dict.items()})
            for summary in summary_list:
                self.summary_writer.add_summary(summary, step)
        if summaries_merged is not None:
            self.summary_writer.add_summary(summaries_merged, step)

    # def load_val_data(self, v2=False):
        # print("Loading Validation data..")
        # self.test_data = {'X': np.load(self.args.data_dir + "X_val.npy"),
                          # 'Y': np.load(self.args.data_dir + "Y_val.npy")}
        # self.test_data = self.resize(self.test_data)
        # self.test_data['Y_large'] = self.test_data['Y']
        # if v2:
            # out_shape = (self.test_data['Y'].shape[1] // self.targets_resize,
                         # self.test_data['Y'].shape[2] // self.targets_resize)
            # yy = np.zeros((self.test_data['Y'].shape[0], out_shape[0], out_shape[1]), dtype=self.test_data['Y'].dtype)
            # for y in range(self.test_data['Y'].shape[0]):
                # yy[y, ...] = misc.imresize(self.test_data['Y'][y, ...], out_shape, interp='nearest')
            # self.test_data['Y'] = yy

        # self.test_data_len = self.test_data['X'].shape[0] - self.test_data['X'].shape[0] % self.args.batch_size
        # print("Validation-shape-x -- " + str(self.test_data['X'].shape))
        # print("Validation-shape-y -- " + str(self.test_data['Y'].shape))
        # self.num_iterations_testing_per_epoch = (self.test_data_len + self.args.batch_size - 1) // self.args.batch_size
        # print("Validation data is loaded")

    def test_generator(self):
        start = 0
        new_epoch_flag = True
        idx = None
        while True:
            # init index array if it is a new_epoch
            if new_epoch_flag:
                if self.args.shuffle:
                    idx = np.random.choice(self.test_data_len, self.test_data_len, replace=False)
                else:
                    idx = np.arange(self.test_data_len)
                new_epoch_flag = False

            # select the mini_batches
            mask = idx[start:start + self.args.batch_size]
            x_batch = self.test_data['X'][mask]
            y_batch = self.test_data['Y'][mask]

            # update start idx
            start += self.args.batch_size

            if start >= self.test_data_len:
                start = 0
                new_epoch_flag = True

            yield x_batch, y_batch


    def test(self, pkl=False):
        print("Testing mode will begin NOW..")

        # load the best model checkpoint to test on it
        self.load_best_model()

        # load mini_batches
        # x_batch = self.test_data['X'][idx:idx + 1]
        # y_batch = self.test_data['Y'][idx:idx + 1]

        # cap = cv2.VideoCapture(1)
        rospy.Subscriber("/camera/image_color", Image, self.video_callback)

        while not rospy.is_shutdown():
            rospy.loginfo("Ready to run network.")
        # while True:
            # start = time.time()
            # # ret, frame = cap.read()

            # # if ret == True:
            # # frame = frame[0:512, 0:1024]
            # x_batch = self.img

            # feed_dict = {self.test_model.x_pl: x_batch,
                        # # self.test_model.y_pl: y_batch,
                        # self.test_model.is_training: False}

            # # run the feed_forward
            # out_argmax = self.sess.run([self.test_model.out_argmax],
                                                        # # self.test_model.segmented_summary],
                                                        # feed_dict=feed_dict)

            # print(out_argmax.shape)
            # # cv2.imshow('segmentation', segmented_imgs[0])
            # # if cv2.waitKey(1) & 0xFF == ord('q'):
                # # break

            # print("Total time: ", time.time() - start)
            rospy.spin()

    def video_callback(self, data):
        """
        Callback for the ROS image topic.
        """
        start = time.time()
        height = data.height
        width = data.width
        img = np.ndarray(shape=(height, width, 3),
                              dtype='uint8', buffer=data.data)

        # Crop Image
        # img = img[511:-1, 168:1192, :]

        # Add dimension for network to recognize
        img = np.expand_dims(img, 0)

        x_batch = img

        feed_dict = {self.test_model.x_pl: x_batch,
                    # self.test_model.y_pl: y_batch,
                    self.test_model.is_training: False}

        # run the feed_forward
        # out_argmax, seg_img = self.sess.run([self.test_model.out_argmax,
                                            # self.test_model.segmented_summary],
                                            # feed_dict=feed_dict)
        out_argmax = self.sess.run([self.test_model.out_argmax],
                                    feed_dict=feed_dict)

        # seg_img = self.sess.run([self.test_model.segmented_summary],
                                            # feed_dict=feed_dict)

        self.build_and_pub(out_argmax)

        print("Total time: ", time.time() - start)

    def build_and_pub(self, seg_img):
        print(type(seg_img))
        print(seg_img[0].shape)
        img = np.squeeze(seg_img[0], 0)
        img = 10 * np.array(img, dtype='uint8')

        img_msg = Image()
        img_msg.header.frame_id = 'camera'
        img_msg.header.stamp = rospy.Time.now()
        img_msg.height = img.shape[0]
        img_msg.width = img.shape[1]
        img_msg.encoding = 'mono8'
        img_msg.is_bigendian = False
        img_msg.data = img.tobytes()
        img_msg.step = img.shape[1]

        self.seg_pub.publish(img_msg)

    def finalize(self):
        self.reporter.finalize()
        self.summary_writer.close()
        self.save_model()

