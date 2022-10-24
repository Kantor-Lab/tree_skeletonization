

import open3d as o3d
import glob
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path

class multiway_registration:
    def __init__(self, data_path, method, voxel_size, filter_nb_points, filter_radius, normal_radius, normal_max_nn):
        self.data_path = data_path
        self.method = method
        self.voxel_size = voxel_size
        self.filter_nb_points = filter_nb_points
        self.filter_radius = filter_radius
        self.normal_radius = normal_radius
        self.normal_max_nn = normal_max_nn
        self.max_correspondence_distance_coarse = self.voxel_size * 15
        self.max_correspondence_distance_fine = self.voxel_size * 1.5
        self.dense_pointclouds = self._load_pointclouds_from_folder()
        self.sparse_pointclouds = self._preprocess_pointcloud()
        self.pose_graph = o3d.pipelines.registration.PoseGraph()
        
    def _load_pointclouds_from_folder(self):
        pc_paths = sorted(glob.glob(os.path.join(self.data_path, 'PC_world_'+self.method, '*.ply'), recursive=True))
        dense_pointclouds = []
        for pc_path in pc_paths:
            point_cloud = o3d.io.read_point_cloud(pc_path)
            dense_pointclouds.append(point_cloud) 
        return dense_pointclouds  

    def _preprocess_pointcloud(self):
        '''
        Voxelize, filter, and calculate normal. 
        '''
        sparse_pointclouds = []
        for pointcloud in self.dense_pointclouds:
            #Voxelize
            pointcloud = pointcloud.voxel_down_sample(voxel_size=self.voxel_size)
            # Filter 
            _, ind = pointcloud.remove_radius_outlier(nb_points=self.filter_nb_points, radius=self.filter_radius)
            pointcloud = pointcloud.select_by_index(ind)
            pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_radius, max_nn=self.normal_max_nn))
            pointcloud.orient_normals_to_align_with_direction(orientation_reference=[-1, 0, 0])
            #o3d.visualization.draw_geometries([pointcloud])  # test     
            sparse_pointclouds.append(pointcloud)
        return sparse_pointclouds

    def _pairwise_registration(self, source, target):
        print("Apply point-to-plane ICP")
        icp_coarse = o3d.pipelines.registration.registration_icp(
            source, target, self.max_correspondence_distance_coarse, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        icp_fine = o3d.pipelines.registration.registration_icp(
            source, target, self.max_correspondence_distance_fine,
            icp_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        transformation_icp = icp_fine.transformation
        information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, self.max_correspondence_distance_fine,
            icp_fine.transformation)
        return transformation_icp, information_icp

    def _full_registration(self):
        odometry = np.identity(4)
        self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
        n_pcds = len(self.sparse_pointclouds)
        for source_id in range(n_pcds):
            for target_id in range(source_id + 1, n_pcds):
                transformation_icp, information_icp = self._pairwise_registration(self.sparse_pointclouds[source_id], self.sparse_pointclouds[target_id])
                print("Build o3d.pipelines.registration.PoseGraph")
                if target_id == source_id + 1:  # odometry case
                    odometry = np.dot(transformation_icp, odometry)
                    self.pose_graph.nodes.append(
                        o3d.pipelines.registration.PoseGraphNode(
                            np.linalg.inv(odometry)))
                    self.pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                target_id,
                                                                transformation_icp,
                                                                information_icp,
                                                                uncertain=False))
                else:  # loop closure case
                    self.pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                target_id,
                                                                transformation_icp,
                                                                information_icp,
                                                                uncertain=True))
        

    def optimize_pose_graph(self):
        print("Full registration ...")
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Error) as cm:
            self._full_registration()
            
        print("Optimizing PoseGraph ...")
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.max_correspondence_distance_fine,
            edge_prune_threshold=0.25,
            reference_node=0)
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Error) as cm:
            o3d.pipelines.registration.global_optimization(
                self.pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option)
        for point_id in range(len(self.dense_pointclouds)):
            self.dense_pointclouds[point_id].transform(self.pose_graph.nodes[point_id].pose)

    
    def generate_pointclouds(self, out_name, out_voxel_size, out_filter_nb_points, out_filter_radius):
        dense_pcd_combined = o3d.geometry.PointCloud()
        for point_id in range(len(self.dense_pointclouds)):
            dense_pcd_combined += self.dense_pointclouds[point_id]
        # Save point cloud from multiway registration
        dense_pcd_combined = dense_pcd_combined.voxel_down_sample(voxel_size=out_voxel_size)
        _, ind = dense_pcd_combined.remove_radius_outlier(nb_points=out_filter_nb_points, radius=out_filter_radius)
        dense_pcd_combined = dense_pcd_combined.select_by_index(ind)

        Path(os.path.join(self.data_path, 'PC_REGISTERED')).mkdir(parents=True, exist_ok=True)
        out_path = Path(os.path.join(self.data_path, 'PC_REGISTERED', str(out_name)+'_'+self.method+'.ply'))
        o3d.io.write_point_cloud(str(out_path), dense_pcd_combined)
        print(dense_pcd_combined)
        dense_pcd_combined.clear()



