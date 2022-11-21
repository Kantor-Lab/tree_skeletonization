import open3d as o3d
import numpy as np
import copy
import sys
import time
import argparse
import os
from modules.treeskel.skeleton import make_cloud, skeletonize

sys.setrecursionlimit(4000)


if __name__ == '__main__':

    t_start = time.time()
    parser = argparse.ArgumentParser(description='Compute skeleton and evaluate.')
    parser.add_argument('method', type=str, default='default', help='default / field / mst / ftsem')
    parser.add_argument('tree_id', type=int, default=1, help='tree ID')
    parser.add_argument('leaf_density', type=int, default=1, help='leaf density')
    args = parser.parse_args()

    if args.method == 'field':
        filename = f'assets/field/tree_{args.tree_id}_preproc.npy'
    else:
        filename = f'assets/simulation/tree_{args.tree_id}_{args.leaf_density}_preproc.npy'
    tree_preproc = np.load(filename, allow_pickle=True)

    voxel_size = tree_preproc.item().get('voxel_size')
    main_tree_pcd, likelihood_map_pcd, tree_mesh, _ = skeletonize(
        method=args.method,
        edges=tree_preproc.item().get('edges'),
        radius=tree_preproc.item().get('radius'),
        likelihood_values=tree_preproc.item().get('likelihood_values'),
        likelihood_points=tree_preproc.item().get('likelihood_points'),
        likelihood_colors=tree_preproc.item().get('likelihood_colors'),
        voxel_size=voxel_size,
    )

    skeleton_gt_pcd = None
    if args.method != 'field':
        skeleton_gt = tree_preproc.item().get('skeleton_gt')
        skeleton_gt_color = np.zeros_like(skeleton_gt)
        skeleton_gt_color[:, 2] = 1
        skeleton_gt_pcd = make_cloud(skeleton_gt, skeleton_gt_color)

    out_dir = 'output/{}/tree_{}_{}'.format(args.method, args.tree_id, args.leaf_density)
    os.makedirs(out_dir, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(out_dir, 'observed.ply'),
                             make_cloud(tree_preproc.item().get('raw_pcd_points'),
                                        tree_preproc.item().get('raw_pcd_colors')))
    o3d.io.write_point_cloud(os.path.join(out_dir, 'predicted.ply'), main_tree_pcd)
    if skeleton_gt_pcd is not None:
        o3d.io.write_point_cloud(os.path.join(out_dir, 'groundtruth.ply'), skeleton_gt_pcd)
    if likelihood_map_pcd is not None:
        o3d.io.write_point_cloud(os.path.join(out_dir, 'likelihood.ply'), likelihood_map_pcd)
    if tree_mesh is not None:
        o3d.io.write_triangle_mesh(os.path.join(out_dir, 'sphere_tree.ply'), tree_mesh)
    computation_time = time.time() - t_start
    with open('output/{}/results.txt'.format(args.method), 'a') as file:
        file.write('{} {} {} {}\n'.format(args.tree_id, args.leaf_density, voxel_size, computation_time))
