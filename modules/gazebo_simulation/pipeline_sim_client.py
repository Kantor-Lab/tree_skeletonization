#!/usr/bin/env python

import rospy
from pointcloud_segmentation.srv import pipeline_sim
import numpy as np
import tf


def pipeline_sim_client(request, transform):
    rospy.wait_for_service('pipeline_sim')
    try:
        service_call = rospy.ServiceProxy('pipeline_sim', pipeline_sim)
        response = service_call(request, transform)
        return response.response
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


if __name__ == "__main__":
    rospy.init_node('test_client', anonymous=True) #initialize the node 
    
    tf_listener = tf.TransformListener()        
    tf_listener.waitForTransform('/world', '/depth', rospy.Time(), rospy.Duration(4.0))
    (trans, q) = tf_listener.lookupTransform('/world', '/depth', rospy.Time(0)) 
    T = tf.transformations.quaternion_matrix(q)
    T[:3,3]=np.array(trans)
    T = T.reshape(-1)
    print(pipeline_sim_client(0, T))
