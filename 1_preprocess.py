#!/usr/bin/env python

import sys
import open3d as o3d
import numpy as np
import time
import os
import cv2
import click
from tqdm import tqdm
from joblib import Parallel, delayed

from modules.instance_segmentation import detectron
from modules.treeskel import edge_extractor
from modules.treeskel.likelihood_map import construct_likelihood_map
from modules.treeskel import multiway_registration
sys.setrecursionlimit(30000)

class FieldDataPreprocessor:
    """
    A class used to preprocess field data for 3D point cloud analysis.

    Attributes
    ----------
    pcd_combined : open3d.geometry.PointCloud
        Combined point cloud from all frames.
    edges_combined : list
        List of edges combined from all frames.
    radius_combined : list
        List of radii combined from all frames.
    scores_combined : list
        List of scores combined from all frames.
    detectron : object
        Detectron2 predictor for instance segmentation.
    """

    def __init__(self):
        """
        Initializes the FieldDataPreprocessor with default parameters.
        """
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
        """
        Converts RGB-D images to a point cloud.

        Parameters
        ----------
        color_img : np.ndarray
            The color image.
        depth_img : np.ndarray
            The depth image.
        camera_info : object
            Camera information containing intrinsic parameters.

        Returns
        -------
        open3d.geometry.PointCloud
            The resulting point cloud.
        """
        color_img = o3d.geometry.Image(color_img)
        depth_img = o3d.geometry.Image(depth_img)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_img, depth_img, depth_scale=1, convert_rgb_to_intensity=False)
        intrinsic_matrix = np.array(camera_info.K).reshape(3, 3)
        fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(camera_info.width, camera_info.height, fx, fy, cx, cy)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
        return pcd

    @staticmethod
    def clusters_from_masks(masks, left_rectified, disparity_map, camera_info, scores, max_distance=1.0, visualize=False):
        """
        Extracts clusters from masks and reprojects disparity maps to 3D points.

        Parameters
        ----------
        masks : list
            List of binary masks.
        left_rectified : np.ndarray
            Rectified left image.
        disparity_map : np.ndarray
            Disparity map.
        camera_info : object
            Camera information containing intrinsic parameters.
        scores : list
            Scores for each mask.
        max_distance : float, optional
            Maximum distance to consider points (default is 1.0).
        visualize : bool, optional
            Flag to visualize the clusters (default is False).

        Returns
        -------
        tuple
            Combined point cloud, cluster indices, and scores per cluster.
        """
        stereo_baseline = -0.059634685
        camera_focal_length_px = camera_info.K[0, 0]
        image_center_w = camera_info.K[0, 2]
        image_center_h = camera_info.K[1, 2]

        Q = np.float32([
            [1, 0, 0, -image_center_w],
            [0, 1, 0, -image_center_h],
            [0, 0, 0, camera_focal_length_px],
            [0, 0, -1 / stereo_baseline, 0]
        ])

        points = cv2.reprojectImageTo3D(disparity_map, Q)
        combined_pcd = o3d.geometry.PointCloud()
        cluster_idx = 0
        cluster_indices = []
        new_scores_per_cluster = []

        for i, (mask, score) in enumerate(zip(masks, scores)):
            p_idx = np.asarray(np.where(mask)).T
            cluster_points = points[p_idx[:, 0], p_idx[:, 1]]
            cluster_color = left_rectified[p_idx[:, 0], p_idx[:, 1]]

            valid_mask = (cluster_points[:, 2] > 0) & ~np.isinf(cluster_points).any(axis=1) & (cluster_points[:, 2] < max_distance)
            cluster_points, cluster_color = cluster_points[valid_mask], cluster_color[valid_mask]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cluster_points)
            pcd.colors = o3d.utility.Vector3dVector(cluster_color / 255)

            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.01)
            pcd, _ = pcd.remove_radius_outlier(nb_points=20, radius=0.005)

            if len(pcd.points) < 100:
                continue

            # TODO: This is a bottleneck
            labels = np.array(pcd.cluster_dbscan(eps=0.01, min_points=3))

            if len(labels) == 0 or labels.max() < 0:
                continue

            for label in range(labels.max() + 1):
                inliers = np.argwhere(labels == label).flatten()
                pcd_dbscan = pcd.select_by_index(inliers)

                if visualize:
                    colors = np.random.rand(len(pcd_dbscan.points), 3)
                    pcd_dbscan.colors = o3d.utility.Vector3dVector(colors)

                if len(pcd_dbscan.points) < 200:
                    continue

                cluster_indices.extend([cluster_idx] * len(pcd_dbscan.points))
                combined_pcd += pcd_dbscan
                new_scores_per_cluster.append(score)
                cluster_idx += 1

        if visualize:
            o3d.visualization.draw_geometries([combined_pcd])

        return combined_pcd, cluster_indices, np.array(new_scores_per_cluster)

    @staticmethod
    def merge_masks(branch_masks, scores):
        """
        Merges overlapping masks based on a predefined overlap percentage.

        Parameters
        ----------
        branch_masks : list
            List of branch masks.
        scores : list
            Scores for each mask.

        Returns
        -------
        tuple
            Merged masks and scores.
        """
        flag = True
        while flag:
            break_flag = False
            num_masks = len(branch_masks)
            for i in range(num_masks):
                for j in range(num_masks):
                    if i == j:
                        continue
                    overlapping_pixels = np.sum(branch_masks[i] * branch_masks[j])
                    overlap_percentage = max(overlapping_pixels / np.sum(branch_masks[i]), overlapping_pixels / np.sum(branch_masks[j]))
                    if overlap_percentage > 0.7:
                        branch_masks[j] = branch_masks[i] + branch_masks[j]
                        branch_masks = np.delete(branch_masks, i, 0)
                        scores = np.delete(scores, i)
                        break_flag = True
                        break
                if break_flag:
                    break
            if not break_flag:
                flag = False
        return branch_masks, scores

    def process_frame(self, frame_id, data_dict, icp_tf=None):
        """
        Processes a single frame of data.

        Parameters
        ----------
        frame_id : int
            Frame identifier.
        data_dict : dict
            Dictionary containing frame data.
        icp_tf : np.ndarray, optional
            Initial transformation matrix (default is None).
        """
        visualize = False

        color_img = data_dict['color']
        depth_img = data_dict['depth'][:1080, :1440]
        transformation_matrix = data_dict['tf']
        camera_info = data_dict['camera_info']

        depth_img = np.expand_dims(depth_img, axis=2)
        if frame_id > 33:
            transformation_matrix = np.dot(icp_tf, transformation_matrix)

        branch_masks, scores = self.detectron.predict(color_img, visualize=visualize)
        branch_masks, scores = self.merge_masks(branch_masks, scores)
        pcd, cluster_indices, scores = self.clusters_from_masks(branch_masks, color_img, depth_img, camera_info, scores, max_distance=0.8, visualize=visualize)

        if len(cluster_indices) == 0:
            return

        edges, radius, scores = edge_extractor.extract_nurbs(pcd, cluster_indices, scores, ctrlpts_size=4, degree=1, visualize=visualize)

        edges_homo = np.ones((edges.reshape(-1, 3).shape[0], 4))
        edges_homo[:, :3] = edges.reshape(-1, 3)
        edges_homo = edges_homo @ transformation_matrix.T
        self.edges_combined.append(edges_homo[:, :3])
        self.radius_combined.append(radius)
        self.scores_combined.append(scores)

        pcd = pcd.transform(transformation_matrix)
        self.pcd_combined += pcd

def pointcloud_registration(data_list):
    """
    Registers a list of point clouds.

    Parameters
    ----------
    data_list : list
        List of data dictionaries for each frame.

    Returns
    -------
    open3d.geometry.PointCloud
        The registered point cloud.
    """
    voxel_size = 0.002
    filter_nb_points = 50
    filter_radius = 0.02
    normal_radius = 0.02
    normal_max_nn = 20
    out_voxel_size = 0.001
    out_filter_nb_points = 200
    out_filter_radius = 0.01
    reg = multiway_registration.multiway_registration(data_list, voxel_size=voxel_size, filter_nb_points=filter_nb_points, filter_radius=filter_radius, normal_radius=normal_radius, normal_max_nn=normal_max_nn)
    registered_pcd = reg.generate_pointclouds(out_voxel_size, out_filter_nb_points, out_filter_radius)
    return registered_pcd

def pairwise_registration_p2p(source, target, tf_init=np.identity(4)):
    """
    Performs pairwise point-to-point registration using ICP.

    Parameters
    ----------
    source : open3d.geometry.PointCloud
        Source point cloud.
    target : open3d.geometry.PointCloud
        Target point cloud.
    tf_init : np.ndarray, optional
        Initial transformation matrix (default is identity matrix).

    Returns
    -------
    np.ndarray
        Transformation matrix after ICP.
    """
    voxel_size = 0.002
    source = source.voxel_down_sample(voxel_size)
    source, _ = source.remove_radius_outlier(nb_points=50, radius=voxel_size * 3)
    target = target.voxel_down_sample(voxel_size)
    target, _ = target.remove_radius_outlier(nb_points=50, radius=voxel_size * 3)

    threshold = 0.01
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, tf_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=3000))
    return reg_p2p.transformation

def branch_filter(pcd):
    """
    Filters point cloud to retain only branch points based on color.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        Point cloud to be filtered.

    Returns
    -------
    open3d.geometry.PointCloud
        Filtered point cloud with only branch points.
    """
    colors = np.array(pcd.colors)
    points = np.array(pcd.points)
    mask = np.all((colors[:, 1] < colors[:, 0], colors[:, 2] < colors[:, 0]), axis=0)
    points, colors = points[mask], colors[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

@click.command()
@click.option('--data_npy', type=str, default='real_tree_9.npy', help='Path to the npz file containing the data')
@click.option('--voxel_size', type=float, default=0.002, help='Voxel size for the likelihood map')
def main(data_npy, voxel_size):
    """
    Main function to process the data and construct the likelihood map.

    Parameters
    ----------
    data_npy : str
        Path to the npz file containing the data.
    voxel_size : float
        Voxel size for the likelihood map.
    """
    data_npy = os.path.abspath(data_npy)
    data_dir = os.path.dirname(data_npy)
    file_name = os.path.splitext(os.path.basename(data_npy))[0]
    data_list = list(np.load(data_npy, allow_pickle=True))
    print(f'Processing {data_npy} with {len(data_list)} frames')

    t = time.time()
    print('Registering pointclouds, this may take a while (~30 minutes)...')
    bot_pcd_raw = pointcloud_registration(data_list[:33])
    top_pcd_raw = pointcloud_registration(data_list[34:])
    bot_pcd_branch = branch_filter(bot_pcd_raw)
    top_pcd_branch = branch_filter(top_pcd_raw)
    icp_tf = pairwise_registration_p2p(top_pcd_branch, bot_pcd_branch)
    print(f'ICP took {time.time() - t:.3f} s')

    print('Initializing preprocessor...')
    preprocessor = FieldDataPreprocessor()
    t_start = time.time()
    print('Processing frames...')
    for i, data_dict in enumerate(tqdm(data_list)):
        preprocessor.process_frame(i, data_dict, icp_tf)
    print(f'Frame processing took {time.time() - t_start:.3f} s')

    edges_combined = np.concatenate(preprocessor.edges_combined).reshape(-1, 2, 3)
    radius_combined = np.concatenate(preprocessor.radius_combined)
    scores_combined = np.concatenate(preprocessor.scores_combined)

    t = time.time()
    likelihood_pcd, likelihood = construct_likelihood_map(edges_combined, radius_combined, scores_combined, voxel_size=voxel_size, visualize=False)
    print(f'Likelihood map construction took {time.time() - t:.3f} s')
    computation_time = time.time() - t_start

    tree_preprocessed = {
        'edges': edges_combined,
        'radius': radius_combined,
        'raw_pcd_points': np.asarray(preprocessor.pcd_combined.points),
        'raw_pcd_colors': np.asarray(preprocessor.pcd_combined.colors),
        'likelihood_values': np.asarray(likelihood),
        'likelihood_points': np.asarray(likelihood_pcd.points),
        'likelihood_colors': np.asarray(likelihood_pcd.colors),
        'voxel_size': voxel_size
    }

    np.save(os.path.join(data_dir, f'{file_name}_preprocessed.npy'), tree_preprocessed)
    print(f'{data_npy} preprocessing took {computation_time:.3f} s')

if __name__ == "__main__":
    main()
