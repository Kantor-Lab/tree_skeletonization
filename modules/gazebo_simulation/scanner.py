#!/usr/bin/env python

import os
import copy
import rospy
import numpy as np
import math
#from datetime import datetime
import tf 
import moveit_commander
#from rosbag_record import RosbagRecord
#from camera_trigger import Camera
#from apriltag_detector import ApriltagDetector
from pointcloud_segmentation.srv import pipeline_sim


class scanner:
    def __init__(self, group, robot, dist_trunk, scan_radius, scan_angle, num_levels, level_offset, velocity=0.1):
        self.group = group
        self.group.set_max_velocity_scaling_factor(velocity)
        self.robot = robot
        self.velocity=velocity
        #date = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        #self.data_dir = os.path.join(os.path.dirname( __file__ ), '..', 'data', date)
        #os.mkdir(self.data_dir)
        self.joint_home = [0.7243904296923747, -1.2968201333753893, -2.550225401144985, 0.705816896704377, 0.8455071727333356, 0]
        self.pose_home = None
        self.trunk_axis = np.array([0,0,1]) # This depends on the world frame gravity axis
        self.dist_trunk = dist_trunk
        self.scan_radius = scan_radius
        self.scan_angle = scan_angle # degrees
        self.num_levels = num_levels
        self.level_offset = level_offset

        self.ee_joint = 5 # 6 on real robot
        #self.camera = Camera()
        #self.april = ApriltagDetector()
        self.tf_listener = tf.TransformListener()        

    def safe_set_pose_target(self, pose_target):
        #set pose_target as the goal pose of manipulator group 
        self.group.set_pose_target(pose_target) 

        # Wait for user input to go to scan position
        while True:
            plan = self.group.plan() #call plan function to plan the path
            val = raw_input('Confirm planned path for home position to execute: (Y/n)')
            if val=='Y':
                self.group.execute(plan, wait=True) #execute plan on real/simulation robot
                self.group.stop() # Calling `stop()` ensures that there is no residual movement
                self.group.clear_pose_targets()
                break
            elif val=='x':
                moveit_commander.roscpp_shutdown() 
                raise KeyboardInterrupt

    def go_home(self):
        self.group.set_joint_value_target(self.joint_home) 
        plan = self.group.plan() 
        self.group.execute(plan, wait=True) 
        self.group.stop()        
        self.pose_home = self.group.get_current_pose()
        
    def set_home(self):
        pose_current = self.group.get_current_pose()
        pose_current = self.group.get_current_pose()
        
        # orient orthogonal down
        px = pose_current.pose.position.x
        py = pose_current.pose.position.y
        pz = pose_current.pose.position.z
        qx = pose_current.pose.orientation.x
        qy = pose_current.pose.orientation.y
        qz = pose_current.pose.orientation.z
        qw = pose_current.pose.orientation.w

        # Transformation matrix from pose        
        T = tf.transformations.quaternion_matrix([qx, qy, qz, qw])
        T[:3,3]=np.array([px, py, pz])
        print('current:')
        print(T)
        T[:3,:3] = np.round(T[:3, :3])
        print('target:')
        print(T)
        # Convert to quaternion
        q_target = tf.transformations.quaternion_from_matrix(T)

        # Update target pose
        pose_target = copy.copy(pose_current)
        pose_target.pose.orientation.x = q_target[0]
        pose_target.pose.orientation.y = q_target[1]
        pose_target.pose.orientation.z = q_target[2]
        pose_target.pose.orientation.w = q_target[3]
        
        self.safe_set_pose_target(pose_target)
    
        # set joint 6 to 0 radians
        joint_target = self.group.get_current_joint_values() 
        joint_target[self.ee_joint] = 0
        self.group.set_joint_value_target(joint_target) 
        plan = self.group.plan() 
        self.group.execute(plan, wait=True) 
        self.group.stop()
        
        self.joint_home = copy.copy(joint_target)
        print(self.joint_home)
        self.pose_home = self.group.get_current_pose()
        print('Scan position reached')
		

    def scan(self, plan, tree_id, num_viewpoints):
        #cmd = 'rosbag record /joint_states /tf /tf_static /theia/left/camera_info /theia/left/image_raw /theia/right/camera_info /theia/right/image_raw'
        #cmd = 'rosbag record /joint_states /tf'
        #rosbag_save_path = os.path.join(self.data_dir, '{}.bag'.format(tree_id))
        #rosbag_record = RosbagRecord(cmd, rosbag_save_path)
        rospy.sleep(1)
        self.execute_plan_by_joints(plan, num_viewpoints)
        #self.group.set_joint_value_target(self.joint_home_bot) 
        #self.group.go(wait=True) 
        #self.group.stop()
        rospy.sleep(1)
        #rosbag_record.stop_record()

    def execute_waypoints(self, plan):
        retimed_plan = self.group.retime_trajectory(self.robot.get_current_state(), plan, self.velocity) # Retime trajectory with scaled velocity
        self.group.execute(retimed_plan, wait=True)
        self.group.stop()        
        # Go Home
        self.group.set_joint_value_target(self.joint_home) 
        plan = self.group.plan() 
        self.group.execute(plan, wait=True) 
        self.group.stop()
    
    def execute_plan_by_joints(self, plan, num_viewpoints):
        num_waypoints = len(plan.joint_trajectory.points)
        step = int(num_waypoints/num_viewpoints)
        joint_waypoints = []
        for i in range(num_waypoints):
            joint_waypoints.append(plan.joint_trajectory.points[i].positions)
        count = 0
        for i in range(0, num_waypoints, step):
            self.group.set_joint_value_target(joint_waypoints[i]) 
            self.group.go(wait=True) 
            self.group.stop()
            rospy.sleep(1)
            self.tf_listener.waitForTransform('/world', '/depth', rospy.Time(), rospy.Duration(4.0))
            (trans, q) = self.tf_listener.lookupTransform('/world', '/depth', rospy.Time(0)) 
            T = tf.transformations.quaternion_matrix(q)
            T[:3,3]=np.array(trans)
            T = T.reshape(-1)
            response = pipeline_sim_client(count, T)
            count+=1
            #self.camera.trigger()
        self.group.set_joint_value_target(joint_waypoints[0]) 
        self.group.go(wait=True) 
        self.group.stop()
        rospy.sleep(1)
        #self.camera.trigger()

            
        # Go Home
        #self.group.set_joint_value_target(self.joint_home) 
        #plan = self.group.plan() 
        #self.group.execute(plan, wait=True) 
        #self.group.stop()

    def generate_cylindrical_waypoints(self):
        '''Tilt primitive motion of robot. 
        Parameters:
            point (list): 3-D coordinate of point in rotation axis
            axis (list): 3-D vector of rotation axis (right-hand rule)
            angle (double): angle of tilting (degrees)
            velocity (double): robot velocity between 0 and 1
        Returns:
        
        '''
        
        
        # Pose variables. The parameters can be seen from "$ rosmsg show Pose"
        init_pose = self.group.get_current_pose().pose 
        pos_initial = [init_pose.position.x, init_pose.position.y, init_pose.position.z]
        ori_initial = [init_pose.orientation.x, init_pose.orientation.y, init_pose.orientation.z, init_pose.orientation.w]

        point = np.array(pos_initial) + np.array([self.dist_trunk,0,0]) 

        # Tilt center point. Closest point from tcp to axis line    
        center = np.add(point, np.dot(np.subtract(pos_initial, point), self.trunk_axis)*self.trunk_axis)

        # Closest distance from tcp to axis line
        radius = self.scan_radius #np.linalg.norm(np.subtract(center, pos_initial))
        
        # Pair of orthogonal vectors in tilt plane
        v1 =  -np.subtract(np.add(center, np.dot(np.subtract(pos_initial, center), self.trunk_axis)*self.trunk_axis), pos_initial)
        v1 = v1/np.linalg.norm(v1)
        v2 = np.cross(self.trunk_axis, v1)

        # Interpolate orientation poses via quaternion slerp
        q_left = axis_angle2quaternion(self.trunk_axis, self.scan_angle/2)
        q_right = axis_angle2quaternion(self.trunk_axis, -self.scan_angle/2)
        
        q_left =  tf.transformations.quaternion_multiply(q_left, ori_initial)    
        q_right =  tf.transformations.quaternion_multiply(q_right, ori_initial)    
        
        q_waypoints = slerp(q_right, q_left, np.arange(0 , 1.0+1.0/self.scan_angle, 1.0/self.scan_angle)) 

        pose_target = self.group.get_current_pose().pose 
        angle_range = range(-self.scan_angle/2, self.scan_angle/2+1)
        waypoints = []
        base_waypoints = []
        assert(len(angle_range)==len(q_waypoints))
        for i, angle in enumerate(angle_range):
            circle = np.add(center, radius*(math.cos(math.radians(angle)))*v1 + radius*(math.sin(math.radians(angle)))*v2)
            pose_target.position.x = circle[0]
            pose_target.position.y = circle[1] 
            pose_target.position.z = circle[2] 
            pose_target.orientation.x = q_waypoints[i][0]
            pose_target.orientation.y = q_waypoints[i][1]
            pose_target.orientation.z = q_waypoints[i][2]
            pose_target.orientation.w = q_waypoints[i][3]
            base_waypoints.append(copy.deepcopy(pose_target))
        it = range(len(base_waypoints))

        home_pose = self.group.get_current_pose().pose 
        waypoints.append(copy.deepcopy(home_pose))
        for l in range(self.num_levels):
            for i in it:
                waypoint = copy.deepcopy(base_waypoints[i])
                if self.trunk_axis[0]:
                    waypoint.position.x+=self.level_offset*l
                elif self.trunk_axis[1]:
                    waypoint.position.y+=self.level_offset*l
                elif self.trunk_axis[2]:
                    waypoint.position.z+=self.level_offset*l
                waypoints.append(copy.deepcopy(waypoint))
            it.reverse()
        (plan, fraction) = self.group.compute_cartesian_path(waypoints, 0.01, 0) # waypoints, resolution=1cm, jump_threshold)
        print('Waypoints followed: {}'.format(fraction))
        return (plan, fraction)

    
    def generate_spherical_waypoints_from_tag(self, heights_ratio):

    	# TODO
        radius = self.scan_radius
        sample_heights = np.array(heights_ratio)*radius
        num_points_per_circle = 100

        # Pose variables. The parameters can be seen from "$ rosmsg show Pose"
        init_pose = self.group.get_current_pose().pose 
        pos_init = [init_pose.position.x, init_pose.position.y, init_pose.position.z]
        q_init = [init_pose.orientation.x, init_pose.orientation.y, init_pose.orientation.z, init_pose.orientation.w]


        # April stuff
        T_april_world = self.april.detect_tag()

        p_ref_april = np.array([0,0,radius,1])
        p_ref_world = np.matmul(T_april_world, p_ref_april)
        
        dx = p_ref_world[0] - T_april_world[0,3] 
        dy = p_ref_world[1] - T_april_world[1,3] 
        dz = p_ref_world[2] - T_april_world[2,3] 

        v = -np.array([dx,dy,dz])/np.linalg.norm([dx,dy,dz])
        u = np.array([1,0,0])
        n = -np.cross(v,u)
        n = n/np.linalg.norm(n)
        alpha = np.arccos(np.dot(v,u))

        qw = np.cos(alpha/2)
        qx = np.sin(alpha/2)*n[0]
        qy = np.sin(alpha/2)*n[1]
        qz = np.sin(alpha/2)*n[2]
        q_ref_world = tf.transformations.quaternion_multiply(q_init, [qx, qy, qz, qw])    


          
        #T_ref = tf.transformations.quaternion_matrix(q_init)
        #T_ref[:3,3]=np.array(pos_init)
        T_ref = tf.transformations.quaternion_matrix(q_ref_world)
        T_ref[:3,3]=np.array(p_ref_world[:3])
        
        T_target = np.eye(4) # Frame of apriltag w.r.t. ee_link
        T_target[0,3] = radius

        sample_radius = np.sqrt(radius**2-sample_heights**2)
        step = 2*np.pi/num_points_per_circle
        pose_waypoints_homo = []
        q_waypoints = []
        for i, r in enumerate(sample_radius):
            for theta in np.arange(0, 2*np.pi+step, step):
                y = r*np.cos(theta)
                z = r*np.sin(theta)
                x = -sample_heights[i]
                pose_waypoints_homo.append([x,y,z,1])

                v = -np.array([x,y,z])/np.linalg.norm([x,y,z])
                u = np.array([1,0,0])
                n = -np.cross(v,u)
                n = n/np.linalg.norm(n)
                alpha = np.arccos(np.dot(v,u))

                qw = np.cos(alpha/2)
                qx = np.sin(alpha/2)*n[0]
                qy = np.sin(alpha/2)*n[1]
                qz = np.sin(alpha/2)*n[2]
                #q_world = tf.transformations.quaternion_multiply(q_init, [qx, qy, qz, qw])    
                q_world = tf.transformations.quaternion_multiply(q_ref_world, [qx, qy, qz, qw])    
                q_waypoints.append(q_world)


        pose_waypoints_homo = np.array(pose_waypoints_homo)
        pose_waypoints_homo_world = np.matmul(T_ref, np.matmul(T_target, pose_waypoints_homo.T)).T
        

        waypoints = []
        waypoints.append(copy.deepcopy(init_pose))
        pose_target = self.group.get_current_pose().pose 
        pose_target.position.x = p_ref_world[0]
        pose_target.position.y = p_ref_world[1]
        pose_target.position.z = p_ref_world[2]
        pose_target.orientation.x = q_ref_world[0]
        pose_target.orientation.y = q_ref_world[1]
        pose_target.orientation.z = q_ref_world[2]
        pose_target.orientation.w = q_ref_world[3]
        waypoints.append(copy.deepcopy(pose_target))
        for i, pose in enumerate(pose_waypoints_homo_world):
            pose_target.position.x = pose[0]
            pose_target.position.y = pose[1] 
            pose_target.position.z = pose[2] 
            pose_target.orientation.x = q_waypoints[i][0]
            pose_target.orientation.y = q_waypoints[i][1]
            pose_target.orientation.z = q_waypoints[i][2]
            pose_target.orientation.w = q_waypoints[i][3]
            waypoints.append(copy.deepcopy(pose_target))
        #waypoints.append(copy.deepcopy(init_pose))
        (plan, fraction) = self.group.compute_cartesian_path(waypoints, 0.01, 0) # waypoints, resolution=1cm, jump_threshold)
        print('Waypoints followed: {}'.format(fraction))
        return (plan, fraction)

    def generate_spherical_waypoints(self, heights_ratio):
        
        radius = self.scan_radius
        sample_heights = np.array(heights_ratio)*radius
        num_points_per_circle = 100

        # Pose variables. The parameters can be seen from "$ rosmsg show Pose"
        init_pose = self.group.get_current_pose().pose 
        pos_init = [init_pose.position.x, init_pose.position.y, init_pose.position.z]
        q_init = [init_pose.orientation.x, init_pose.orientation.y, init_pose.orientation.z, init_pose.orientation.w]

          
        T_ref = tf.transformations.quaternion_matrix(q_init)
        T_ref[:3,3]=np.array(pos_init)
        
        T_target = np.eye(4) # Frame of apriltag w.r.t. ee_link
        T_target[0,3] = radius

        sample_radius = np.sqrt(radius**2-sample_heights**2)
        step = 2*np.pi/num_points_per_circle
        pose_waypoints_homo = []
        q_waypoints = []
        for i, r in enumerate(sample_radius):
            for theta in np.arange(0, 2*np.pi+step, step):
                y = r*np.cos(theta)
                z = r*np.sin(theta)
                x = -sample_heights[i]
                pose_waypoints_homo.append([x,y,z,1])

                v = -np.array([x,y,z])/np.linalg.norm([x,y,z])
                u = np.array([1,0,0])
                n = -np.cross(v,u)
                n = n/np.linalg.norm(n)
                alpha = np.arccos(np.dot(v,u))

                qw = np.cos(alpha/2)
                qx = np.sin(alpha/2)*n[0]
                qy = np.sin(alpha/2)*n[1]
                qz = np.sin(alpha/2)*n[2]
                q_world = tf.transformations.quaternion_multiply(q_init, [qx, qy, qz, qw])    
                q_waypoints.append(q_world)


        pose_waypoints_homo = np.array(pose_waypoints_homo)
        pose_waypoints_homo_world = np.matmul(T_ref, np.matmul(T_target, pose_waypoints_homo.T)).T
        

        pose_target = self.group.get_current_pose().pose 
        waypoints = []
        for i, pose in enumerate(pose_waypoints_homo_world):
            pose_target.position.x = pose[0]
            pose_target.position.y = pose[1] 
            pose_target.position.z = pose[2] 
            pose_target.orientation.x = q_waypoints[i][0]
            pose_target.orientation.y = q_waypoints[i][1]
            pose_target.orientation.z = q_waypoints[i][2]
            pose_target.orientation.w = q_waypoints[i][3]
            waypoints.append(copy.deepcopy(pose_target))
        #waypoints.append(copy.deepcopy(init_pose))
        (plan, fraction) = self.group.compute_cartesian_path(waypoints, 0.01, 0) # waypoints, resolution=1cm, jump_threshold)
        print('Waypoints followed: {}'.format(fraction))
        return (plan, fraction)

def slerp(v0, v1, t_array):
    # >>> slerp([1,0,0,0],[0,0,0,1],np.arange(0,1,0.001))
    t_array = np.array(t_array)
    v0 = np.array(v0)
    v1 = np.array(v1)
    dot = np.sum(v0*v1)

    if (dot < 0.0):
        v1 = -v1
        dot = -dot
    
    DOT_THRESHOLD = 0.9995
    if (dot > DOT_THRESHOLD):
        result = v0[np.newaxis,:] + t_array[:,np.newaxis]*(v1 - v0)[np.newaxis,:]
        return (result.T / np.linalg.norm(result, axis=1)).T
    
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0*t_array
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0[:,np.newaxis] * v0[np.newaxis,:]) + (s1[:,np.newaxis] * v1[np.newaxis,:])

def axis_angle2quaternion(axis, angle):
    '''Convert axis angle representation to quaternion (assumes axis is already normalised).
    Parameters:
        axis (list): 3-D normalised vector of rotation axis (right-hand rule)
        angle (double): Magnitude of tilt angle in degrees
    Returns:
        quaternion (list): quaternion representation in order of qx, qy, qz, qw
    
    '''
    s = math.sin(math.radians(angle)/2)
    qx = axis[0] * s
    qy = axis[1] * s
    qz = axis[2] * s
    qw = math.cos(math.radians(angle)/2)
    return [qx, qy, qz, qw]

def pipeline_sim_client(request, transform):
    rospy.wait_for_service('pipeline_sim')
    try:
        pipeline_call = rospy.ServiceProxy('pipeline_sim', pipeline_sim)
        response = pipeline_call(request, transform)
        return response.response
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)
