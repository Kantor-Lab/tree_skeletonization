import os 
import glob
import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
import re
from scipy.spatial.transform import Rotation


# https://github.com/BonJovi1/Stereo-Reconstruction/blob/master/code.ipynb
# Takes disparity map and returns o3d.geometry.Pointcloud 
def branchcloud_from_disparity(disparity_map, masks, scores, left_rectified, camera_param_left, stereo_baseline, max_distance=2.0, visualize=False):
    camera_focal_length_px = camera_param_left['intrinsics'][0,0]
    image_center_w = camera_param_left['intrinsics'][0,2]
    image_center_h = camera_param_left['intrinsics'][1,2] 
    image_height, image_width, _ = left_rectified.shape
    #image_width = camera_param_left['width']
    #image_height = camera_param_left['height']

    Q = np.float32([[1, 0,                          0,        -image_center_w],
                    [0, 1,                          0,        -image_center_h], 
                    [0, 0,                          0, camera_focal_length_px], 
                    [0, 0,         -1/stereo_baseline,                      0]])
    
    points = cv2.reprojectImageTo3D(disparity_map, Q)
    offset=0
    points = points[offset:image_height+offset,offset:image_width+offset,:]
    pcd = o3d.geometry.PointCloud()     
    cluster_indices = []   
    for i, mask in enumerate(masks):
        p_idx = np.asarray(np.where(mask)).T 
        cluster_points = points[p_idx[:,0], p_idx[:,1]]
        cluster_color = left_rectified[p_idx[:,0], p_idx[:,1]]

        # Mask 1: Points that have positive depth
        m = cluster_points[:,2]>0
        cluster_points = cluster_points[m]
        cluster_color = cluster_color[m]   
        
        # Mask 2: Points that are not -inf/inf
        m = ~np.isinf(cluster_points).any(axis=1)
        cluster_points = cluster_points[m]
        cluster_color = cluster_color[m]    
        
        # Mask 3: Points that are within max_distance meters
        m = cluster_points[:,2]<max_distance
        cluster_points = cluster_points[m]
        cluster_color = cluster_color[m]  

        pcd_cluster = o3d.geometry.PointCloud()        
        pcd_cluster.points = o3d.utility.Vector3dVector(cluster_points)
        pcd_cluster.colors = o3d.utility.Vector3dVector(cluster_color/255)
        pcd += pcd_cluster
    
        cluster_indices = cluster_indices + [i]*len(cluster_points)
    if visualize:
        o3d.visualization.draw_geometries([pcd])
    return pcd, cluster_indices


def remove_statistical_outlier(pcd, cluster_indices, nb_neighbors=100, std_ratio=0.1):
    #pcd, ind = pcd.remove_radius_outlier(nb_points=200, radius=0.02)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    cluster_indices = [cluster_indices[i] for i in ind]
    return pcd, cluster_indices

# https://github.com/BonJovi1/Stereo-Reconstruction/blob/master/code.ipynb
# Takes disparity map and returns o3d.geometry.Pointcloud 
def semanticloud_from_disparity(disparity_map, label, label_prob, left_rectified, camera_param_left, stereo_baseline, max_distance=2.0):
    camera_focal_length_px = camera_param_left['intrinsics'][0,0]
    image_center_w = camera_param_left['intrinsics'][0,2]
    image_center_h = camera_param_left['intrinsics'][1,2] 
    image_width = camera_param_left['width']
    image_height = camera_param_left['height']

    Q = np.float32([[1, 0,                          0,        -image_center_w],
                    [0, 1,                          0,        -image_center_h], 
                    [0, 0,                          0, camera_focal_length_px], 
                    [0, 0,         -1/stereo_baseline,                      0]])
    
    points = cv2.reprojectImageTo3D(disparity_map, Q)
    
    #branch_mask = np.ma.masked_where(label==0, label, copy=False).mask
    #leaf_mask = np.ma.masked_where(label==1, label, copy=False).mask
    #petal_mask = np.ma.masked_where(label==2, label, copy=False).mask
    background_mask = ~np.ma.masked_where(label==3, label, copy=False).mask
    
    mask = background_mask
    mask_rgb = np.zeros_like(left_rectified)
    mask_rgb[:,:,0] = mask
    mask_rgb[:,:,1] = mask
    mask_rgb[:,:,2] = mask
    offset = 0
    points = points[offset:image_height+offset,offset:image_width+offset,:]*mask_rgb
    
    
    # remove nan points that are behind the camera if it exists
    points_filtered = points.reshape(-1,3)
    colors = left_rectified.reshape(-1,3)
    label = label.reshape(-1)
    label_prob = label_prob.reshape(-1)

    # Mask 1: Points that have positive depth
    mask = points_filtered[:,2]>0
    points_filtered = points_filtered[mask]
    colors = colors[mask]   
    label = label[mask]
    label_prob = label_prob[mask]

    # Mask 2: Points that are not -inf/inf
    mask = ~np.isinf(points_filtered).any(axis=1)
    points_filtered = points_filtered[mask]
    colors = colors[mask]    
    label = label[mask]
    label_prob = label_prob[mask]
    
    # Mask 3: Points that are within max_distance meters
    mask = points_filtered[:,2]<max_distance
    points_filtered = points_filtered[mask]
    colors = colors[mask]        
    label = label[mask]
    label_prob = label_prob[mask]
    
    labels = np.zeros_like(points_filtered)
    labels[:,0] = label
    labels[:,1] = label_prob
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_filtered)
    point_cloud.colors = o3d.utility.Vector3dVector(colors/255)
    point_cloud.normals = o3d.utility.Vector3dVector(labels)
    
    return point_cloud

# https://github.com/BonJovi1/Stereo-Reconstruction/blob/master/code.ipynb
# Takes disparity map and returns o3d.geometry.Pointcloud 
def pointcloud_from_disparity(disparity_map, left_rectified, camera_param_left, stereo_baseline, max_distance=3.0):
    camera_focal_length_px = camera_param_left['intrinsics'][0,0]
    image_center_w = camera_param_left['intrinsics'][0,2]
    image_center_h = camera_param_left['intrinsics'][1,2] 
    image_width = camera_param_left['width']
    image_height = camera_param_left['height']

    Q = np.float32([[1, 0,                          0,        -image_center_w],
                    [0, 1,                          0,        -image_center_h], 
                    [0, 0,                          0, camera_focal_length_px], 
                    [0, 0,         -1/stereo_baseline,                      0]])
    
    points = cv2.reprojectImageTo3D(disparity_map, Q)
    mask = np.sum(left_rectified, axis=2)>50
    mask_rgb = np.zeros_like(left_rectified)
    mask_rgb[:,:,0] = mask
    mask_rgb[:,:,1] = mask
    mask_rgb[:,:,2] = mask
    offset = 0
    points = points[offset:image_height+offset,offset:image_width+offset,:]*mask_rgb
    # remove nan points that are behind the camera if it exists
    points_filtered = points.reshape(-1,3)
    colors = left_rectified.reshape(-1,3)
    
    # Mask 1: Points that have positive depth
    mask = points_filtered[:,2]>0
    points_filtered = points_filtered[mask]
    colors = colors[mask]    
    # Mask 2: Points that are not -inf/inf
    mask = ~np.isinf(points_filtered).any(axis=1)
    points_filtered = points_filtered[mask]
    colors = colors[mask]    
    # Mask 3: Points that are within max_distance meters
    mask = points_filtered[:,2]<max_distance
    points_filtered = points_filtered[mask]
    colors = colors[mask]        
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_filtered)
    point_cloud.colors = o3d.utility.Vector3dVector(colors/255)
    return point_cloud

def pointcloud_from_folder(data_path, method, camera_param_left, stereo_baseline, max_distance=3.0):
    Path(os.path.join(data_path, 'PC_camera_'+method)).mkdir(parents=True, exist_ok=True)
    out_path = Path(os.path.join(data_path, 'PC_camera_'+method))

    disparity_map_paths = sorted(glob.glob(os.path.join(data_path, 'DM_'+method, '*.npy'), recursive=True))
    left_rect_paths = sorted(glob.glob(os.path.join(data_path, 'LEFT_RECT', '*'), recursive=True))
    for (disparity_map_path, left_rect_path) in tqdm(list(zip(disparity_map_paths, left_rect_paths))):
        disparity_map = np.load(disparity_map_path)
        left_rect = cv2.imread(left_rect_path)
        left_rect = cv2.cvtColor(left_rect, cv2.COLOR_BGR2RGB)
        point_cloud = pointcloud_from_disparity(disparity_map, left_rect, camera_param_left, stereo_baseline, max_distance)
        numbers = [int(s) for s in re.findall(r'-?\d+?\d*', left_rect_path)]
        out_dir = str(out_path / 'pcd_camera{:04d}.ply'.format(numbers[-1]))
        o3d.io.write_point_cloud(out_dir, point_cloud)

def pointcloud_worldframe(point_cloud, T_world_camera):
    return point_cloud.transform(T_world_camera)
    
def pointcloud_worldframe_folder(data_path, method, frame_change=None):
    if frame_change is None:
        frame_change = np.eye(4)
    Path(os.path.join(data_path, 'PC_world_'+method)).mkdir(parents=True, exist_ok=True)
    out_path = Path(os.path.join(data_path, 'PC_world_'+method))

    pc_cam_paths = sorted(glob.glob(os.path.join(data_path, 'PC_camera_'+method, '*.ply'), recursive=True))
    tf_paths = sorted(glob.glob(os.path.join(data_path, 'TF', '*.txt'), recursive=True))
    for (pc_cam_path, tf_path) in tqdm(list(zip(pc_cam_paths, tf_paths))):
        pointcloud_camera = o3d.io.read_point_cloud(pc_cam_path)
        cam_pose = np.loadtxt(tf_path)
        cam_pos = cam_pose[:3]
        cam_quat = cam_pose[-4:]
        r_cam = Rotation.from_quat(cam_quat)
        T_world_camera = transformation_from_rotation_translation([r_cam.as_matrix()], [cam_pos])[0]
        T_world_camera = np.matmul(T_world_camera, frame_change)
        pointcloud_world = pointcloud_worldframe(pointcloud_camera, T_world_camera)
        numbers = [int(s) for s in re.findall(r'-?\d+?\d*', pc_cam_path)]
        out_dir = str(out_path / 'pcd_world{:04d}.ply'.format(numbers[-1]))
        o3d.io.write_point_cloud(out_dir, pointcloud_world)

def transformation_from_rotation_translation(rotation, translation):
    transformations = []
    for r, t in zip(rotation,translation):
        T = np.eye(4)
        T[:3, :3] = r
        T[:3, 3] = t
        transformations.append(T)
    return np.array(transformations)
