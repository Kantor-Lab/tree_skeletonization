#!/usr/bin/env python
import os
import numpy as np
import open3d as o3d
import argparse
import time
import tqdm
from modules.treeskel import skeleton as skel

def evaluate(observed_pcd, predicted_pcd, groundtruth_pcd=None):
    if groundtruth_pcd is None:
        observed_pcd = observed_pcd.voxel_down_sample(voxel_size=0.002)
    search_radius=0.02
    min_dist_to_check = 0.002
    cossim_threshold = 0.5 # 0 means perpendicular

    predicted_pcd.paint_uniform_color([0, 0, 0])
    
    observed_tree = o3d.geometry.KDTreeFlann(observed_pcd)
    predicted_tree = o3d.geometry.KDTreeFlann(predicted_pcd)
    predicted_array = np.array(predicted_pcd.points)
    total_prediction = len(predicted_array)
    
    if groundtruth_pcd is not None:
        groundtruth_pcd.paint_uniform_color([0, 0, 0])
        groundtruth_tree = o3d.geometry.KDTreeFlann(groundtruth_pcd)
        groundtruth_array = np.array(groundtruth_pcd.points)
  
    TP_count = 0
    TP_occ_count = 0
    FP_count = 0

    prediction_color_code = []
    if groundtruth_pcd is None:
        for predicted_point in tqdm.tqdm(predicted_array):
            flag_observed= False
            [k_p, idx_p, _] = predicted_tree.search_knn_vector_3d(predicted_point, 3)
            [k_o, idx_o, dist_o] = observed_tree.search_radius_vector_3d(predicted_point, search_radius)
            # Estimate vector of that point by taking the average of two points closest to it
            pd_vector = predicted_array[idx_p[1]]-predicted_array[idx_p[2]]
            pd_unit_vector = pd_vector/np.linalg.norm(pd_vector)
            for i_o, d_o in zip(idx_o, dist_o):
                point_o = np.array(observed_pcd.points)[i_o]
                o_vector = point_o - predicted_point
                o_unit_vector = o_vector/np.linalg.norm(o_vector)
                cossim = np.dot(pd_unit_vector, o_unit_vector)
                min_dist_criteria = np.sqrt(d_o)<min_dist_to_check
                if np.abs(cossim)<cossim_threshold or min_dist_criteria:
                    flag_observed = True
                    break
            if flag_observed:
                TP_count+=1
                prediction_color_code.append([1,0,0]) # TP red
            else:
                TP_occ_count+=1
                prediction_color_code.append([0,1,0]) # TP_occ, green
    else:
        for predicted_point in tqdm.tqdm(predicted_array):
            flag_observed, flag_gt = False, False
            [k_p, idx_p, _] = predicted_tree.search_knn_vector_3d(predicted_point, 3)
            [k_o, idx_o, _] = observed_tree.search_radius_vector_3d(predicted_point, search_radius)
            [k_g, idx_g, dist_g] = groundtruth_tree.search_radius_vector_3d(predicted_point, search_radius)
            # Estimate vector of that point by taking the average of two points closest to it
            pd_vector = predicted_array[idx_p[1]]-predicted_array[idx_p[2]]
            pd_unit_vector = pd_vector/np.linalg.norm(pd_vector)
            for i_g, d_g in zip(idx_g, dist_g):
                point_g = groundtruth_array[i_g]
                g_vector = point_g - predicted_point
                g_unit_vector = g_vector/np.linalg.norm(g_vector)
                cossim = np.dot(pd_unit_vector, g_unit_vector)
                min_dist_criteria = np.sqrt(d_g)<min_dist_to_check
                if np.abs(cossim)<cossim_threshold or min_dist_criteria:
                    flag_gt = True
                    break
            if flag_gt:
                for i_o in idx_o:
                    point_o = np.array(observed_pcd.points)[i_o]
                    o_vector = point_o - predicted_point
                    o_unit_vector = o_vector/np.linalg.norm(o_vector)
                    cossim = np.dot(pd_unit_vector, o_unit_vector)
                    if np.abs(cossim)<cossim_threshold:
                        flag_observed = True
                        break
            if flag_observed:
                TP_count+=1
                prediction_color_code.append([0,0,1]) # TP, blue
            elif flag_gt:
                TP_occ_count+=1
                prediction_color_code.append([0,1,0]) # TP_occ, green
            else:
                FP_count+=1
                prediction_color_code.append([1,0,0]) # FP, red
        accuracy = (TP_count+TP_occ_count)/total_prediction
        print('Accuracy: {}/{} --- {}'.format(TP_count+TP_occ_count, total_prediction, accuracy))
    predicted_pcd.colors = o3d.utility.Vector3dVector(np.array(prediction_color_code))
    OSR = TP_occ_count/total_prediction
    print('OSR: {}/{} --- {}'.format(TP_occ_count, total_prediction, OSR))    
    assert(TP_count+TP_occ_count+FP_count==total_prediction)
    if groundtruth_pcd is None:
        return predicted_pcd, TP_occ_count, TP_count, total_prediction
    else:
        return groundtruth_pcd, predicted_pcd, TP_count, TP_occ_count, FP_count, total_prediction

def get_ground_truth(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    pcd = mesh.sample_points_poisson_disk(number_of_points=30000, init_factor=3)
    min_bound = [0,0,0]
    pcd = pcd.translate(np.array([0,0,1-min_bound[2]]))
    pcd = pcd.rotate(np.array(
        [[-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]]
    ), center=[0,0,0])
    mesh = mesh.translate(np.array([0,0,1-min_bound[2]]))
    mesh = mesh.rotate(np.array(
        [[-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]]
    ), center=[0,0,0])

    search_radius = 0.03
    for iter in range(5):
        pcd = skel.laplacian_smoothing(pcd, search_radius=search_radius)

    graph = skel.UndirectedGraph(pcd, search_radius_scale=50)
    graph.construct_initial_graphs()
    graph.merge_components()
    pcd, _ = graph.distribute_equally(spacing=0.001)
    
    return pcd

if __name__=='__main__':
    t_start = time.time()
    parser = argparse.ArgumentParser(description='Compute skeleton and evaluate.')
    parser.add_argument('method', type=str, default='default', help='default / field / mst / ftsem')
    parser.add_argument('tree_id', type=int, help='tree ID')
    parser.add_argument('leaf_density', type=int, help='leaf density')
    args = parser.parse_args()
    
    out_dir = 'output/evaluation/{}/tree_{}_{}'.format(args.method, args.tree_id, args.leaf_density)
    os.makedirs(out_dir, exist_ok=True)
    observed_pcd = o3d.io.read_point_cloud('output/{}/tree_{}_{}/observed.ply'.format(args.method, args.tree_id, args.leaf_density)) 
    predicted_pcd = o3d.io.read_point_cloud('output/{}/tree_{}_{}/predicted.ply'.format(args.method, args.tree_id, args.leaf_density))    
    if args.method!='field':
        gt_pcd = o3d.io.read_point_cloud('output/{}/tree_{}_{}/groundtruth.ply'.format(args.method, args.tree_id, args.leaf_density))
        groundtruth_pcd, predicted_pcd, TP_count, TP_occ_count, FP_count, total_prediction = evaluate(observed_pcd, predicted_pcd, gt_pcd)
        o3d.io.write_point_cloud(os.path.join(out_dir, 'groundtruth.ply'), gt_pcd)
    else:
        predicted_pcd, TP_occ_count, TP_count, total_prediction = evaluate(observed_pcd, predicted_pcd)
        FP_count = 0
    o3d.io.write_point_cloud(os.path.join(out_dir, 'predicted.ply'), predicted_pcd)
    o3d.io.write_point_cloud(os.path.join(out_dir, 'observed.ply'), observed_pcd)
    
    computation_time = time.time() - t_start
    with open('output/evaluation/{}/results.txt'.format(args.method), 'a') as file1:
        file1.write('{} {} {} {} {} {} {}\n'.format(args.tree_id, args.leaf_density, TP_count, TP_occ_count, FP_count, total_prediction, computation_time))
    print('Computation Time: {}s'.format(computation_time))
