import open3d as o3d
import numpy as np
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
    """
    Main function to process the preprocessed tree data and construct a skeleton.
    
    Parameters
    ----------
    data_npy : str
        Path to the npy file containing the preprocessed data.
    """
    data_npy = os.path.abspath(data_npy)
    data_dir = os.path.dirname(data_npy)
    file_name = os.path.splitext(os.path.basename(data_npy))[0]
    data_preprocessed = np.load(data_npy, allow_pickle=True).item()
    
    edges = data_preprocessed['edges']
    radius = data_preprocessed['radius']
    raw_pcd_points = data_preprocessed['raw_pcd_points']
    raw_pcd_colors = data_preprocessed['raw_pcd_colors']
    likelihood_values = data_preprocessed['likelihood_values']
    likelihood_points = data_preprocessed['likelihood_points']
    likelihood_colors = data_preprocessed['likelihood_colors']
    voxel_size = data_preprocessed['voxel_size']
    
    print(f'Loaded data from {data_npy}')
    t_start = time.time()
    
    # Set voxel size for likelihood map based on memory capacity
    likelihood_pcd = o3d.geometry.PointCloud()
    likelihood_pcd.points = o3d.utility.Vector3dVector(likelihood_points)
    likelihood_pcd.colors = o3d.utility.Vector3dVector(likelihood_colors)

    for voxel_size in np.arange(0.01, 0.1, 0.001):
        likelihood_map_vg = o3d.geometry.VoxelGrid.create_from_point_cloud(likelihood_pcd, voxel_size=voxel_size)
        likelihood_map_points = np.asarray([likelihood_map_vg.origin + pt.grid_index * likelihood_map_vg.voxel_size for pt in likelihood_map_vg.get_voxels()])
        likelihood_map_values = np.asarray([pt.color[0] for pt in likelihood_map_vg.get_voxels()])
        
        if len(likelihood_map_points) < 55000:
            voxel_size = round(voxel_size, 3)
            print(f'Selected Voxel Size {voxel_size} resulting in {len(likelihood_map_points)} likelihood map nodes.')
            break

    # Construct likelihood map tree
    likelihood_map_pcd = o3d.geometry.PointCloud()
    likelihood_map_pcd.points = o3d.utility.Vector3dVector(likelihood_map_points)
    jet_color_map = plt.get_cmap('jet')
    likelihood_map_colors = jet_color_map(likelihood_map_values)[:, :3]
    likelihood_map_pcd.colors = o3d.utility.Vector3dVector(likelihood_map_colors)
    likelihood_map_tree = skel.UndirectedGraph(likelihood_map_pcd)
    likelihood_map_tree.construct_skeleton_graph(voxel_size)

    # Define cost function for traversing the likelihood map tree
    def fn_weight(u, v, d):
        return -np.log((likelihood_map_values[u] + likelihood_map_values[v]) / 2)

    # Construct initial tree from observed points
    t = time.time()
    observed_tree = skel.construct_initial_skeleton(edges, radius)
    print(f'Constructed initial tree in {time.time() - t:.3f} s')

    # Merge the observed tree with the likelihood map to predict occluded parts of the tree
    t = time.time()
    merger = skel.SkeletonMerger(observed_tree, likelihood_map_tree, fn_weight)
    main_tree = merger.get_main_tree()
    print(f'Merged trees in {time.time() - t:.3f} s')

    # Smooth out the skeleton
    main_tree.pcd = skel.point_laplacian_smoothing(main_tree.pcd, search_radius=0.015)
    main_tree.nodes_array = np.array(main_tree.pcd.points)
    main_tree.graph_laplacian_smoothing()

    main_tree_pcd, radius = main_tree.distribute_equally(0.001)
    tree_mesh = skel.generate_sphere_mesh(main_tree_pcd, radius)

    # Set color of main tree
    main_tree_colors = np.zeros_like(main_tree_pcd.points)
    main_tree_colors[:, :2] = 1
    main_tree_pcd.colors = o3d.utility.Vector3dVector(main_tree_colors)

    raw_pcd = o3d.geometry.PointCloud()
    raw_pcd.points = o3d.utility.Vector3dVector(raw_pcd_points)
    raw_pcd.colors = o3d.utility.Vector3dVector(raw_pcd_colors)

    # Save output
    out_dir = os.path.join(data_dir, f'output_{file_name}')
    os.makedirs(out_dir, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(out_dir, 'observed.ply'), raw_pcd)
    o3d.io.write_point_cloud(os.path.join(out_dir, 'predicted.ply'), main_tree_pcd)
    o3d.io.write_point_cloud(os.path.join(out_dir, 'likelihood.ply'), likelihood_map_pcd)
    o3d.io.write_triangle_mesh(os.path.join(out_dir, 'sphere_tree.ply'), tree_mesh)
    print(f'\nTotal Computation Time: {time.time() - t_start:.3f} s')

if __name__ == '__main__':
    main()
