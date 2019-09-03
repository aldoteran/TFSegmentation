#!/usr/bin/env python3

"""
ROS node in python3 for semantic scene sgmentation of a video feed.
The node grabs a sensor_msgs.Image topic being published and runs it
through a MobileNet CNN. Republishes an the segmented image with 22 classes.

Two different outputs are avialable, either an RGB image of the segmented scene
(slow) or the argmax image with 22 different values (not so slow), one for every
class.

Default values for the published and subscribed topics will be used if not
specified via a launchfile.

Disclaimer: The code base for this script was from the original repository
            by MSiam.

Author: Aldo Teran <aldot@kth.se>
"""

from train.basic_train import BasicTrain

import numpy as np
import time
import rospy
from sensor_msgs.msg import Image

#TODO: Get topics as rosparams, add RGB segmentation as option.
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

        self.test_data = None
        self.test_data_len = None
        self.num_iterations_testing_per_epoch = None
        self.img = Image()

        self.seg_pub = rospy.Publisher("segmentation", Image, queue_size=1)

        rospy.init_node('whatever')

    def test(self, pkl=False):
        print("Testing mode will begin NOW..")

        # load the best model checkpoint to test on it
        self.load_best_model()

        rospy.Subscriber("/camera/image_color", Image, self.video_callback)

        while not rospy.is_shutdown():
            rospy.loginfo("Ready to run network.")
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

        # Add dimension for network to recognize
        img = np.expand_dims(img, 0)

        x_batch = img

        feed_dict = {self.test_model.x_pl: x_batch,
                    # self.test_model.y_pl: y_batch,
                    self.test_model.is_training: False}

        # Uncomment for segmented image
        # out_argmax, seg_img = self.sess.run([self.test_model.out_argmax,
                                            # self.test_model.segmented_summary],
                                            # feed_dict=feed_dict)
        out_argmax = self.sess.run([self.test_model.out_argmax],
                                    feed_dict=feed_dict)

        self.build_and_pub(out_argmax)

        print("Segmentation rate: ", 1/(time.time() - start))

    def build_and_pub(self, seg_img):
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
        # self.summary_writer.close()
        self.save_model()

