import sys
sys.path.append('src')

import os
import numpy as np

import camera_calibration_parser 
import rectification
import disparity
import pointcloud

if __name__=='__main__':
    data_path = 'data'
    method = 'RAFT'
    frame_change = np.array([[ 0, 1, 0, 0],
                             [-1, 0, 0, 0],
                             [ 0, 0, 1, 0],
                             [ 0, 0, 0, 1]])

    stereo_baseline = np.loadtxt(os.path.join(data_path, 'stereo_baseline.txt'))
    cam_param_left = camera_calibration_parser.parse_pkl(os.path.join(data_path, 'left.pkl'))
    cam_param_right = camera_calibration_parser.parse_pkl(os.path.join(data_path, 'right.pkl'))
    
    rectification.rectify_folder(data_path, cam_param_left, cam_param_right)
    dp = disparity.disparity(method, data_path)
    dp.generate_folder()
    pointcloud.pointcloud_from_folder(data_path, method, cam_param_left, stereo_baseline, max_distance=3.0)
    pointcloud.pointcloud_worldframe_folder(data_path, method, frame_change=frame_change)

    