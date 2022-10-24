import open3d as o3d
import numpy as np
import copy
import sys
import time
import argparse
import os
import matplotlib.pyplot as plt 
from modules.treeskel import skeleton as skel

sys.setrecursionlimit(4000)

if __name__=='__main__':
    t_start = time.time()
    parser = argparse.ArgumentParser(description='Compute skeleton and evaluate.')
    parser.add_argument('method', type=str, default='default', help='default / field / mst / ftsem')
    parser.add_argument('tree_id', type=int, default=1, help='tree ID')
    parser.add_argument('leaf_density', type=int, default=1, help='leaf density')
    args = parser.parse_args()

    if args.method=='field':
        tree_preproc = np.load('assets/field/tree_{}_preproc.npy'.format(args.tree_id), allow_pickle=True)
    else:
        tree_preproc = np.load('assets/simulation/tree_{}_{}_preproc.npy'.format(args.tree_id, args.leaf_density), allow_pickle=True)

    edges = tree_preproc.item().get('edges')
    radius = tree_preproc.item().get('radius')
    raw_pcd_points = tree_preproc.item().get('raw_pcd_points')
    raw_pcd_colors = tree_preproc.item().get('raw_pcd_colors')
    likelihood_values = tree_preproc.item().get('likelihood_values')
    likelihood_points = tree_preproc.item().get('likelihood_points')
    likelihood_colors = tree_preproc.item().get('likelihood_colors')
    voxel_size = tree_preproc.item().get('voxel_size')
    if args.method!='field':
        skeleton_gt = tree_preproc.item().get('skeleton_gt')
    if args.method in ['default', 'field']:
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
        
        main_tree.pcd = skel.laplacian_smoothing(main_tree.pcd, search_radius=0.015)
        main_tree.nodes_array = np.array(main_tree.pcd.points)
        #main_tree.laplacian_smoothing()

        main_tree_pcd, radius = main_tree.distribute_equally(0.001) #0.0005 # update radius
        tree_mesh = skel.generate_sphere_mesh(main_tree_pcd, radius) # update_radius

    elif args.method=='mst':
        observed_tree = skel.construct_initial_skeleton(edges, radius)    
        main_tree = skel.UndirectedGraph(observed_tree.distribute_equally(0.01)[0], search_radius_scale=2)
        main_tree.construct_initial_graphs()
        main_tree.merge_components()
        main_tree.minimum_spanning_tree()
        main_tree.pcd = skel.laplacian_smoothing(main_tree.pcd, search_radius=0.015)
        main_tree.nodes_array = np.array(main_tree.pcd.points)   
        main_tree.laplacian_smoothing()
        main_tree_pcd = main_tree.distribute_equally(0.001)[0]

    elif args.method=='ftsem':
        observed_tree = skel.construct_initial_skeleton(edges, radius)    
        main_tree = skel.UndirectedGraph(observed_tree.distribute_equally(0.01)[0], search_radius_scale=2)
        main_tree.construct_initial_graphs()
        main_tree.merge_components()
        connected = True
        connection_count=0
        while connected:
            connected = main_tree.breakpoint_connection()
            connection_count+=1
            print('{} breakpoints connected.'.format(connection_count))
        main_tree.laplacian_smoothing()
        main_tree_pcd = main_tree.distribute_equally(0.001)[0]

    main_tree_colors = np.zeros_like(main_tree_pcd.points)
    main_tree_colors[:,:2] = 1
    main_tree_pcd.colors = o3d.utility.Vector3dVector(main_tree_colors)

    raw_pcd = o3d.geometry.PointCloud()
    raw_pcd.points = o3d.utility.Vector3dVector(raw_pcd_points)
    raw_pcd.colors = o3d.utility.Vector3dVector(raw_pcd_colors)

    if args.method!='field':
        skeleton_gt_pcd = o3d.geometry.PointCloud()
        skeleton_gt_pcd.points = o3d.utility.Vector3dVector(skeleton_gt)
        skeleton_gt_color = np.zeros_like(skeleton_gt)
        skeleton_gt_color[:, 2] = 1
        skeleton_gt_pcd.colors = o3d.utility.Vector3dVector(skeleton_gt_color)    

    out_dir = 'output/{}/tree_{}_{}'.format(args.method, args.tree_id, args.leaf_density)
    os.makedirs(out_dir, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(out_dir, 'observed.ply'), raw_pcd)
    o3d.io.write_point_cloud(os.path.join(out_dir, 'predicted.ply'), main_tree_pcd)
    if args.method!='field':
        o3d.io.write_point_cloud(os.path.join(out_dir, 'groundtruth.ply'), skeleton_gt_pcd)
    if args.method in ['default', 'field']:
        o3d.io.write_point_cloud(os.path.join(out_dir, 'likelihood.ply'), likelihood_map_pcd)
        o3d.io.write_triangle_mesh(os.path.join(out_dir, 'sphere_tree.ply'), tree_mesh)
    computation_time = time.time() - t_start
    with open('output/{}/results.txt'.format(args.method), 'a') as file1:
        file1.write('{} {} {} {}\n'.format(args.tree_id, args.leaf_density, voxel_size, computation_time))
    