import open3d as o3d
import numpy as np
import copy
import sys
import time
import click
import os
import matplotlib.pyplot as plt 
from modules.treeskel import skeleton as skel
sys.setrecursionlimit(4000)

@click.command()
@click.option('--data_npy', type=str, default='real_tree_9_preprocessed.npy', help='Path to the npy file containing the data')
def main(data_npy):
    # Parse data directory
    data_npy = os.path.abspath(data_npy)
    data_dir = os.path.dirname(data_npy)    
    file_name = os.path.splitext(os.path.basename(data_npy))[0] # Get name of file without extension
    data_preprocessed = np.load(data_npy, allow_pickle=True)
    edges = data_preprocessed.item().get('edges')
    radius = data_preprocessed.item().get('radius')
    raw_pcd_points = data_preprocessed.item().get('raw_pcd_points')
    raw_pcd_colors = data_preprocessed.item().get('raw_pcd_colors')
    likelihood_values = data_preprocessed.item().get('likelihood_values')
    likelihood_points = data_preprocessed.item().get('likelihood_points')
    likelihood_colors = data_preprocessed.item().get('likelihood_colors')
    voxel_size = data_preprocessed.item().get('voxel_size')
    print(f'Loaded data from {data_npy}')

    t_start = time.time()
    for voxel_size in np.arange(0.01, 0.1, 0.001):
        likelihood_pcd = o3d.geometry.PointCloud()
        likelihood_pcd.points = o3d.utility.Vector3dVector(likelihood_points)
        likelihood_pcd.colors = o3d.utility.Vector3dVector(likelihood_colors)
        likelihood_map_pcd = copy.deepcopy(likelihood_pcd)
        likelihood_map_vals_rgb = np.zeros(np.array(likelihood_map_pcd.colors).shape)
        likelihood_map_vals_rgb[:,0] = likelihood_values
        likelihood_map_pcd.colors = o3d.utility.Vector3dVector(likelihood_map_vals_rgb)
        likelihood_map_vg = o3d.geometry.VoxelGrid.create_from_point_cloud(likelihood_map_pcd, voxel_size=voxel_size)
        likelihood_map_points = np.asarray([likelihood_map_vg.origin + pt.grid_index*likelihood_map_vg.voxel_size for pt in likelihood_map_vg.get_voxels()])
        likelihood_map_values = np.asarray([pt.color for pt in likelihood_map_vg.get_voxels()])
        likelihood_map_values = likelihood_map_values[:,0]
        if len(likelihood_map_points)<55000: # Limited by memory usage.
            voxel_size = round(voxel_size, 3)
            print('Selected Voxel Size {} resulting in {} likelihood map nodes.'.format(voxel_size, len(likelihood_map_points)))
            break
    
    likelihood_map_pcd = o3d.geometry.PointCloud()
    likelihood_map_pcd.points = o3d.utility.Vector3dVector(likelihood_map_points)
    jet_color_map = plt.get_cmap('jet')
    likelihood_map_colors = jet_color_map(likelihood_map_values)[:,:3]
    likelihood_map_pcd.colors = o3d.utility.Vector3dVector(likelihood_map_colors)
    likelihood_map_tree = skel.UndirectedGraph(likelihood_map_pcd)
    likelihood_map_tree.construct_skeleton_graph(voxel_size)
    
    def fn_weight(u,v,d):
        return -np.log((likelihood_map_values[u]+likelihood_map_values[v])/2)

    observed_tree = skel.construct_initial_skeleton(edges, radius)
    merger = skel.SkeletonMerger(observed_tree, likelihood_map_tree, fn_weight)
    main_tree = merger.get_main_tree()
    
    #main_tree.pcd = skel.laplacian_smoothing(main_tree.pcd, search_radius=0.015)
    #main_tree.nodes_array = np.array(main_tree.pcd.points)
    #main_tree.laplacian_smoothing()

    main_tree_pcd, radius = main_tree.distribute_equally(0.001) #0.0005 # update radius
    tree_mesh = skel.generate_sphere_mesh(main_tree_pcd, radius) # update_radius

    main_tree_colors = np.zeros_like(main_tree_pcd.points)
    main_tree_colors[:,:2] = 1
    main_tree_pcd.colors = o3d.utility.Vector3dVector(main_tree_colors)

    raw_pcd = o3d.geometry.PointCloud()
    raw_pcd.points = o3d.utility.Vector3dVector(raw_pcd_points)
    raw_pcd.colors = o3d.utility.Vector3dVector(raw_pcd_colors)

    out_dir = os.path.join(data_dir, f'output_{file_name}')
    os.makedirs(out_dir, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(out_dir, 'observed.ply'), raw_pcd)
    o3d.io.write_point_cloud(os.path.join(out_dir, 'predicted.ply'), main_tree_pcd)
    o3d.io.write_point_cloud(os.path.join(out_dir, 'likelihood.ply'), likelihood_map_pcd)
    o3d.io.write_triangle_mesh(os.path.join(out_dir, 'sphere_tree.ply'), tree_mesh)
    computation_time = time.time() - t_start
    print(f'Computation Time: {computation_time:.3f} s')
    

if __name__=='__main__':
    main()