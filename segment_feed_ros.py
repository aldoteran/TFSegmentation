#!usr/share/bin python3.5
"""
Hopefully will segment a camera feed and
publish the segmented image to ROS.

Author: Aldo Teran <aldot@kth.se>
"""

# Uses the TFSegmentation Agent Class to handle the network
from agent import Agent
import numpy as np
import rospy
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# CUDA stuff
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"




