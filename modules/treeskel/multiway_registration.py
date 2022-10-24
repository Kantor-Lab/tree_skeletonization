

import open3d as o3d
import glob
import os
import numpy as np
from pathlib import Path
import cv2

class multiway_registration:
    def __init__(self, data_dict_list, voxel_size, filter_nb_points, filter_radius, normal_radius, normal_max_nn):
        self.data_dict_list = data_dict_list
        self.voxel_size = voxel_size
        self.filter_nb_points = filter_nb_points
        self.filter_radius = filter_radius
        self.normal_radius = normal_radius
        self.normal_max_nn = normal_max_nn
        self.max_correspondence_distance_coarse = self.voxel_size * 15
        self.max_correspondence_distance_fine = self.voxel_size * 1.5
        self.dense_pointclouds = self._generate_preprocess_pointclouds()
        self.sparse_pointclouds = self._preprocess_pointcloud()
        self.pose_graph = o3d.pipelines.registration.PoseGraph()
    
    def _generate_preprocess_pointclouds(self):
        dense_pointclouds = []
        for data_dict in self.data_dict_list:
            color_img = data_dict['color']
            depth_img = data_dict['depth'][:1080, :1440]
            depth_img = np.expand_dims(depth_img, axis=2)
            transformation_matrix = data_dict['tf']
            camera_info = data_dict['camera_info']
            pcd = self._pointcloud_from_disparity(color_img, depth_img, camera_info, transformation_matrix, max_distance=0.8)
            dense_pointclouds.append(pcd) 
        return dense_pointclouds  

    def _pointcloud_from_disparity(self, left_rectified, disparity_map, camera_info, tf, max_distance, visualize=False):
        stereo_baseline = -0.059634685
        camera_focal_length_px = camera_info.K[0,0]
        image_center_w = camera_info.K[0,2]
        image_center_h = camera_info.K[1,2] 

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
        points = points*mask_rgb
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

        # Mask 4: Points that are Blue
        mask = np.invert(np.all((colors[:,0]<colors[:,2], colors[:,1]<colors[:,2]), axis=0))
        points_filtered = points_filtered[mask]
        colors = colors[mask]  

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_filtered)
        pcd.colors = o3d.utility.Vector3dVector(colors/255)
        pcd = pcd.transform(tf)

        if visualize:
            o3d.visualization.draw_geometries([pcd])

        return pcd


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

    def _pairwise_registration(self, source, target, tf_init=np.identity(4)):
        #print("Apply point-to-plane ICP")
        icp_coarse = o3d.pipelines.registration.registration_icp(
            source, target, self.max_correspondence_distance_coarse, tf_init,
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


    def _pairwise_registration_identity(self, source, target, tf_init=np.identity(4)):
        #print("Apply point-to-plane ICP")
        transformation_icp = tf_init
        information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, self.max_correspondence_distance_fine,
            tf_init)
        return transformation_icp, information_icp


    def _pairwise_registration_color(self, source, target, tf_init=np.identity(4)):
        voxel_radius = [0.01, 0.005, 0.0025]
        max_iter = [100, 50, 30]
        current_transformation = tf_init
        #print("3. Colored point cloud registration")
        for scale in range(3):
            iter = max_iter[scale]
            radius = voxel_radius[scale]
            #print([iter, radius, scale])

            #print("3-1. Downsample with a voxel size %.2f" % radius)
            source_down = source.voxel_down_sample(radius)
            target_down = target.voxel_down_sample(radius)

            #print("3-2. Estimate normal.")
            source_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
            source_down.orient_normals_to_align_with_direction(orientation_reference=[-1, 0, 0])
            
            target_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
            target_down.orient_normals_to_align_with_direction(orientation_reference=[-1, 0, 0])
            
            #print("3-3. Applying colored point cloud registration")

            result_icp = o3d.pipelines.registration.registration_colored_icp(
                source_down, target_down, radius, current_transformation,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                relative_rmse=1e-6,
                                                                max_iteration=iter))
            current_transformation = result_icp.transformation
        information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, self.max_correspondence_distance_fine,
            current_transformation)
        return current_transformation, information_icp

    def _full_registration(self):
        odometry = np.identity(4)
        #transformation_icp = np.identity(4)
        self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
        n_pcds = len(self.sparse_pointclouds)
        for source_id in range(n_pcds):
            for target_id in range(source_id + 1, n_pcds):
                print('Full Registration: {}-{}'.format(source_id, target_id))
                try:
                    transformation_icp, information_icp = self._pairwise_registration(self.sparse_pointclouds[source_id], self.sparse_pointclouds[target_id])
                except RuntimeError as e:
                    transformation_icp, information_icp = self._pairwise_registration(self.sparse_pointclouds[source_id], self.sparse_pointclouds[target_id])
                #print("Build o3d.pipelines.registration.PoseGraph")
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

    
    def generate_pointclouds(self, out_voxel_size, out_filter_nb_points, out_filter_radius):
        dense_pcd_combined = o3d.geometry.PointCloud()
        for point_id in range(len(self.dense_pointclouds)):
            dense_pcd_combined += self.dense_pointclouds[point_id]
        # Save point cloud from multiway registration
        dense_pcd_combined = dense_pcd_combined.voxel_down_sample(voxel_size=out_voxel_size)
        _, ind = dense_pcd_combined.remove_radius_outlier(nb_points=out_filter_nb_points, radius=out_filter_radius)
        dense_pcd_combined = dense_pcd_combined.select_by_index(ind)
        
        return dense_pcd_combined
        #Path(os.path.join(self.data_path, 'PC_REGISTERED')).mkdir(parents=True, exist_ok=True)
        #out_path = Path(os.path.join(self.data_path, 'PC_REGISTERED', str(out_name)+'_'+self.method+'.ply'))
        #o3d.io.write_point_cloud(str(out_path), dense_pcd_combined)
        #print(dense_pcd_combined)
        #dense_pcd_combined.clear()



