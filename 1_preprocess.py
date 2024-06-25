#!/usr/bin/env python

import sys
import open3d as o3d
import numpy as np
import time
import os 
import cv2
import click 

from modules.instance_segmentation import detectron 
from modules.treeskel import edge_extractor
from modules.treeskel.likelihood_map import construct_likelihood_map
from modules.treeskel import multiway_registration
sys.setrecursionlimit(30000)

class FieldDataPreprocessor:
    def __init__(self):
        self.pcd_combined = o3d.geometry.PointCloud()
        self.edges_combined = []
        self.radius_combined = []
        self.scores_combined = []
        
        self.detectron = detectron.detectron_predictor(
            config_file='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
            weights_path='model_weights/detectron_branch_segmentation.pth',
            num_classes=1, 
            score_threshold=0.9
        )

    @staticmethod
    def rgbd_to_pcd(color_img, depth_img, camera_info):
        '''
        color_img, depth_img: (w,h,c)
        '''
        color_img = o3d.geometry.Image(color_img)
        depth_img = o3d.geometry.Image(depth_img)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_img, depth_img, depth_scale=1, convert_rgb_to_intensity = False)
        intrinsic_matrix = np.array(camera_info.K).reshape(3,3)
        fx = intrinsic_matrix[0,0]
        fy = intrinsic_matrix[1,1]
        cx = intrinsic_matrix[0,2]
        cy = intrinsic_matrix[1,2]
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(camera_info.width, camera_info.height, fx, fy, cx, cy)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
        return pcd

    @staticmethod
    def clusters_from_masks(masks, left_rectified, disparity_map, camera_info, scores, max_distance=1.0, visualize=False):
        stereo_baseline = -0.059634685
        camera_focal_length_px = camera_info.K[0,0]
        image_center_w = camera_info.K[0,2]
        image_center_h = camera_info.K[1,2] 
        #image_height, image_width, _ = left_rectified.shape

        Q = np.float32([[1, 0,                          0,        -image_center_w],
                        [0, 1,                          0,        -image_center_h], 
                        [0, 0,                          0, camera_focal_length_px], 
                        [0, 0,         -1/stereo_baseline,                      0]])
        
        points = cv2.reprojectImageTo3D(disparity_map, Q)
        #offset=0
        #points = points[offset:image_height+offset,offset:image_width+offset,:]
        combined_pcd = o3d.geometry.PointCloud()    
        cluster_idx = 0 
        cluster_indices = []   
        new_scores_per_cluster = []
        for i, (mask, score) in enumerate(zip(masks, scores)):

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

            pcd = o3d.geometry.PointCloud()        
            pcd.points = o3d.utility.Vector3dVector(cluster_points)
            pcd.colors = o3d.utility.Vector3dVector(cluster_color/255)

            pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.01) # HYPER PARAM
            pcd, ind = pcd.remove_radius_outlier(nb_points=20, radius=0.005) # HYPER PARAM

            if len(pcd.points)<100:
                continue
            labels = np.array(pcd.cluster_dbscan(eps=0.01, min_points=3)) # HYPER PARAM
            if len(labels)==0:
                continue
            max_label = labels.max()
            if max_label<0:
                continue
            elif max_label>0: 
                for label in range(max_label+1):
                    inliers = np.argwhere(labels==label)
                    pcd_dbscan = pcd.select_by_index(inliers)
                    if visualize:
                        colors = np.zeros_like(pcd_dbscan.colors)
                        colors[:] = np.random.rand(3)
                        pcd_dbscan.colors = o3d.utility.Vector3dVector(colors)
                    if len(pcd_dbscan.points)<200: # HERE JOHN
                        continue
                    cluster_indices = cluster_indices + [cluster_idx]*len(np.array(pcd_dbscan.points))
                    combined_pcd += pcd_dbscan  
                    new_scores_per_cluster.append(score)
                    cluster_idx+=1  
            else:
                # For coloring clusters
                if visualize:
                    colors = np.zeros_like(pcd.colors)
                    colors[:] = np.random.rand(3)
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                cluster_indices = cluster_indices + [cluster_idx]*len(np.array(pcd.points))
                combined_pcd += pcd    
                new_scores_per_cluster.append(score)
                cluster_idx+=1
        if visualize:
            o3d.visualization.draw_geometries([combined_pcd])
        return combined_pcd, cluster_indices, np.array(new_scores_per_cluster)

    @staticmethod
    def merge_masks(branch_masks, scores):
        flag = True
        while flag:
            break_flag = False
            num_masks = len(branch_masks)
            for i in range(num_masks):
                for j in range(num_masks):
                    if i==j:
                        continue
                    overlapping_pixels = np.sum(branch_masks[i]*branch_masks[j]) 
                    overlap_percentage = max(overlapping_pixels/np.sum(branch_masks[i]), overlapping_pixels/np.sum(branch_masks[j]))
                    if overlap_percentage>0.7:
                        branch_masks[j] = branch_masks[i]+branch_masks[j]
                        branch_masks = np.delete(branch_masks, i, 0)
                        scores = np.delete(scores, i)
                        break_flag = True
                        break
                if break_flag:
                    break  
            if not break_flag:
                flag=False
        return branch_masks, scores
        
    def process_frame(self, frame_id, data_dict, icp_tf=None):
        visualize=False

        color_img = data_dict['color']
        depth_img = data_dict['depth'][:1080, :1440]
        transformation_matrix = data_dict['tf']
        camera_info = data_dict['camera_info']

        depth_img = np.expand_dims(depth_img, axis=2)
        if frame_id > 33:
            transformation_matrix = np.dot(icp_tf, transformation_matrix)

        t = time.time()
        branch_masks, scores = self.detectron.predict(color_img, visualize=visualize)
        print(f'Instance segmentation: {time.time()-t:.3f} s')

        t = time.time()
        branch_masks, scores = self.merge_masks(branch_masks, scores)
        print(f'Merge masks: {time.time()-t:.3f} s')

        t = time.time()
        pcd, cluster_indices, scores = self.clusters_from_masks(branch_masks, color_img, depth_img, camera_info, scores, max_distance=0.8, visualize=visualize)
        print(f'Cluster extraction: {time.time()-t:.3f} s')
        
        if len(cluster_indices)==0:
            return
            
        t = time.time()
        edges, radius, scores = edge_extractor.extract_nurbs(pcd, cluster_indices, scores, ctrlpts_size=4, degree=1, visualize=visualize)
        print(f'Edge extraction: {time.time()-t:.3f} s')
        
        edges_homo = np.ones((edges.reshape(-1,3).shape[0], 4))
        edges_homo[:,:3] = edges.reshape(-1,3) 
        edges_homo = edges_homo@transformation_matrix.T
        self.edges_combined.append(edges_homo[:,:3])
        self.radius_combined.append(radius)
        self.scores_combined.append(scores)
        
        ##edge_extractor.plot_edges(edges, likelihood_pcd+pcd)

        pcd = pcd.transform(transformation_matrix)
        self.pcd_combined+=pcd
    
def pointcloud_registration(data_list):
    
    voxel_size = 0.002
    filter_nb_points = 50
    filter_radius = 0.02
    normal_radius = 0.02
    normal_max_nn = 20
    out_voxel_size = 0.001
    out_filter_nb_points = 200 #800
    out_filter_radius = 0.01
    reg = multiway_registration.multiway_registration(data_list, voxel_size=voxel_size, filter_nb_points=filter_nb_points, filter_radius=filter_radius, normal_radius=normal_radius, normal_max_nn=normal_max_nn)
    #reg.optimize_pose_graph()
    registered_pcd = reg.generate_pointclouds(out_voxel_size, out_filter_nb_points, out_filter_radius)
    return registered_pcd

def pairwise_registration_p2p(source, target, tf_init=np.identity(4)):

    voxel_size = 0.002
    source = source.voxel_down_sample(voxel_size)
    _, ind = source.remove_radius_outlier(nb_points=50, radius=voxel_size*3)
    source = source.select_by_index(ind)

    target = target.voxel_down_sample(voxel_size)
    _, ind = target.remove_radius_outlier(nb_points=50, radius=voxel_size*3)
    target = target.select_by_index(ind)

    threshold = 0.01
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, tf_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=3000))
    transformation_icp = reg_p2p.transformation       
    return transformation_icp


def branch_filter(pcd):
    colors = np.array(pcd.colors)
    points = np.array(pcd.points)

    # Mask 5: Only keep red points
    mask = np.all((colors[:,1]<colors[:,0], colors[:,2]<colors[:,0]), axis=0)
    points = points[mask]
    colors = colors[mask]  

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

@click.command()
@click.option('--data_npy', type=str, default='real_tree_9.npy', help='Path to the npz file containing the data')
@click.option('--voxel_size', type=float, default=0.002, help='Voxel size for the likelihood map')
def main(data_npy, voxel_size):
    # Parse data directory
    data_npy = os.path.abspath(data_npy)
    data_dir = os.path.dirname(data_npy)    
    file_name = os.path.splitext(os.path.basename(data_npy))[0] # Get name of file without extension
    data_list = list(np.load(data_npy, allow_pickle=True))
    print(f'Processing {data_npy} with {len(data_list)} frames')

    # This part is specific to the UMASS apple orchard dataset, 
    # because the robot switched configurations at frame 33
    t = time.time()
    bot_pcd_raw = pointcloud_registration(data_list[:33])
    top_pcd_raw = pointcloud_registration(data_list[34:])   
    bot_pcd_branch = branch_filter(bot_pcd_raw)
    top_pcd_branch = branch_filter(top_pcd_raw)
    icp_tf = pairwise_registration_p2p(top_pcd_branch, bot_pcd_branch)
    print(f'ICP took {time.time()-t:.3f} s')

    print('Intializing preprocessor...')
    preprocessor = FieldDataPreprocessor()
    t_start = time.time()
    for i, data_dict in enumerate(data_list):
        print(f'\nProcessing frame: {i}/{len(data_list)}')
        preprocessor.process_frame(i, data_dict, icp_tf)
    
    edges_combined = np.concatenate(preprocessor.edges_combined).reshape(-1,2,3)
    radius_combined = np.concatenate(preprocessor.radius_combined)
    scores_combined = np.concatenate(preprocessor.scores_combined)

    likelihood_pcd, likelihood = construct_likelihood_map(edges_combined, radius_combined, scores_combined, voxel_size=voxel_size, visualize=False) 
    computation_time = time.time() - t_start
    
    tree_preprocessed = {}
    tree_preprocessed['edges'] = edges_combined
    tree_preprocessed['radius'] = radius_combined
    tree_preprocessed['raw_pcd_points'] = np.asarray(preprocessor.pcd_combined.points)
    tree_preprocessed['raw_pcd_colors'] = np.asarray(preprocessor.pcd_combined.colors)
    tree_preprocessed['likelihood_values'] = np.asarray(likelihood)
    tree_preprocessed['likelihood_points'] = np.asarray(likelihood_pcd.points)
    tree_preprocessed['likelihood_colors'] = np.asarray(likelihood_pcd.colors)
    tree_preprocessed['voxel_size'] = voxel_size

    np.save(os.path.join(data_dir, f'{file_name}_preprocessed.npy'), tree_preprocessed)
    print(f'{data_npy} preprocessing took {computation_time:.3f} s')

if __name__ == "__main__":
    main()




