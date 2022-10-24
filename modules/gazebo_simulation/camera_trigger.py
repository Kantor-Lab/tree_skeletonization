import rospy 
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from rospy.numpy_msg import numpy_msg
#from cv_bridge import CvBridge

class Camera:
    def __init__(self):
        self.color_img_topic = '/depth_camera/color/image_raw'
        self.depth_img_topic = '/depth_camera/depth/image_raw'
        self.depth_camera_info_topic = '/depth_camera/color/camera_info'
        self.camera_info_msg = rospy.wait_for_message(self.depth_camera_info_topic, CameraInfo, timeout=1)

    def get_color_depth(self):
        print('snap...')
        for i in range(50):
            color_msg = rospy.wait_for_message(self.color_img_topic, Image, timeout=1)
            depth_msg = rospy.wait_for_message(self.depth_img_topic, Image, timeout=1)
        
        color_img = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(color_msg.height, color_msg.width, -1)
        depth_img = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(depth_msg.height, depth_msg.width, -1)

        #color_img_test = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='passthrough')
        #depth_img_test = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        return color_img, depth_img, self.camera_info_msg

if __name__=='__main__':
    rospy.init_node('camera_test_node')
    camera = Camera()
    camera.get_color_depth()
    
