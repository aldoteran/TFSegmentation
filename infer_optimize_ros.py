import tensorflow as tf
import numpy as np
import time
import rospy
from PIL import Image as Img
from sensor_msgs.msg import Image
import argparse
from tqdm import tqdm
from utils.average_meter import FPSMeter

def video_callback(data):
    height = data.height
    width = data.width

    img = np.ndarray(shape=(height, width, 3),
                     dtype='uint8', buffer=data.data)
    img = img[0:360,0:640,:]
    img = np.expand_dims(img, 0)
    out = sess.run(y, feed_dict={x: img, is_training: False})
    out = np.squeeze(out, 0)
    print(out)
    image = Img.fromarray(out, 'P')
    image.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference test")
    parser.add_argument('--graph', type=str)
    parser.add_argument('--iterations', default=1000, type=int)

    # Parse the arguments
    args = parser.parse_args()

    if args.graph is not None:
        with tf.gfile.GFile(args.graph, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
    else:
        raise ValueError("--graph should point to the input graph file.")

    G = tf.Graph()

    with tf.Session(graph=G) as sess:
        # The inference is done till the argmax layer on the logits, as the softmax layer is not important.
        y, = tf.import_graph_def(graph_def, return_elements=['network/output/ArgMax:0'])
        print('Operations in Graph:')
        print([op.name for op in G.get_operations()])
        x = G.get_tensor_by_name('import/network/input/Placeholder:0')
        is_training = G.get_tensor_by_name('import/network/input/Placeholder_2:0'),

        tf.global_variables_initializer().run()

        rospy.init_node("mierda")
        pub = rospy.Publisher('segmentation', Image, queue_size=1)
        rospy.Subscriber("/camera/image_color", Image, video_callback)
        while not rospy.is_shutdown():
            rospy.spin()

