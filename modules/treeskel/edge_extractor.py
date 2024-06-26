import numpy as np
import open3d as o3d
from geomdl import fitting
from ..helper.visualization import LineMesh
import time

def plot_edges(edges, pcd, radius=0.001, vis_kwargs=None):
    edges = np.array(edges)
    edges = np.concatenate(edges)
    edges = edges.reshape(-1,3)
    edge_points = []
    nodes = []
    for i in range(0, len(edges), 2):
        if np.linalg.norm(edges[i]-edges[i+1])>0:
            edge_points.append(edges[i])
            edge_points.append(edges[i+1])
        else:
            nodes.append(edges[i])
    edge_points = np.array(edge_points)
    nodes = np.array(nodes)
    
    nodes_pcd = o3d.geometry.PointCloud()
    nodes_pcd.points = o3d.utility.Vector3dVector(nodes)
    nodes_colors = np.zeros_like(nodes)
    nodes_colors[:,0] = 1
    nodes_pcd.colors = o3d.utility.Vector3dVector(nodes_colors)

    edge_lines = np.array([[i,i+1] for i in range(0, len(edge_points), 2)])
    edge_colors = [[1, 0, 0] for i in range(len(edge_lines))]
    line_mesh = LineMesh(edge_points, edge_lines, edge_colors, radius=radius).cylinder_segments 
    o3d.visualization.draw_geometries([*line_mesh])#, **vis_kwargs)
    if pcd is not None:
        o3d.visualization.draw_geometries([*line_mesh, pcd, nodes_pcd])


def extract_nurbs(pcd, cluster_indices, scores, ctrlpts_size=4, degree=1, visualize=False):
    ctrlpts_size = 4
    edges = []
    radius = []
    radius_edges = []
    new_scores = []
    for cluster_idx in range(max(cluster_indices)+1):
        cluster_by_index = np.argwhere(np.array(cluster_indices)==cluster_idx)
        if len(cluster_by_index)<5:
            continue
        cluster_pcd = pcd.select_by_index(cluster_by_index)
        cluster_array = np.asarray(cluster_pcd.points)
        cluster_mean = np.mean(cluster_array, axis=0)
        cluster_mean_vector_normalized = cluster_mean/np.linalg.norm(cluster_mean)
        cluster_centered = cluster_array - cluster_mean
        covariance_matrix = (cluster_centered.T@cluster_centered)/len(cluster_centered)
        u, s, vh = np.linalg.svd(covariance_matrix)

        cluster_centered_aligned = cluster_centered@u
        mask = cluster_centered_aligned[:,0] < 0.005 # 0.002
        cluster_centered_aligned = cluster_centered_aligned[mask]
        
        mask = cluster_centered_aligned[:,0] > -0.005 # -0.002
        cluster_centered_aligned = cluster_centered_aligned[mask]
        
        if len(cluster_centered_aligned)==0:
            continue

        y_max = np.max(cluster_centered_aligned[:,1])
        y_min = np.min(cluster_centered_aligned[:,1])
        estimated_radius = (y_max-y_min)/2
        center_offset = cluster_mean_vector_normalized*estimated_radius # Radius offset

        radius_edge_centered_aligned = np.array([[0, y_max, 0], [0, y_min, 0]])
        radius_edge_centered = radius_edge_centered_aligned@vh
        radius_edge = radius_edge_centered + cluster_mean
        radius_edges.append(radius_edge)
        
        AR = np.sqrt(s[1]/s[0])
        if AR<0.5:
            cluster_centered_aligned = cluster_centered@u
            cluster_centered_aligned_sorted = cluster_centered_aligned[cluster_centered_aligned[:, 0].argsort()]
            cluster_centered_sorted = cluster_centered_aligned_sorted@vh
            cluster_sorted = cluster_centered_sorted + cluster_mean
            # TODO: Bottleneck here (fitting.approximate_curve) in terms of computation time
            curve = fitting.approximate_curve(cluster_sorted.tolist(), degree=degree, ctrlpts_size=ctrlpts_size)
            curve_points = np.array(curve.ctrlpts)
            for i in range(ctrlpts_size-1):
                edge = [curve_points[i]+center_offset, curve_points[i+1]+center_offset]             
                edges.append(edge)
                radius.append(estimated_radius)
                new_scores.append(scores[cluster_idx])
        else:
            edge_centered_aligned = np.array([[0, 0, 0], [0, 0, 0]])
            edge_centered = edge_centered_aligned@vh
            edge = edge_centered + cluster_mean + center_offset
            edges.append(edge)
            radius.append(estimated_radius)
            new_scores.append(scores[cluster_idx])
    edges = np.asarray(edges)
    radius = np.asarray(radius)
    radius_edges = np.asarray(radius_edges)
    new_scores = np.asarray(new_scores)
    
    if visualize:
        edge_points = edges.reshape(-1,3)
        edge_lines = np.array([[i,i+1] for i in range(0, len(edge_points), 2)])
        edge_colors = [[1, 0, 0] for i in range(len(edge_lines))]
        line_mesh = LineMesh(edge_points, edge_lines, edge_colors, radius=0.002).cylinder_segments 

        radius_edge_points = radius_edges.reshape(-1,3)
        radius_edge_lines = np.array([[i,i+1] for i in range(0, len(radius_edge_points), 2)])
        radius_edge_colors = [[0, 0, 1] for i in range(len(radius_edge_lines))]
        radius_line_mesh = LineMesh(radius_edge_points, radius_edge_lines, radius_edge_colors, radius=0.0005).cylinder_segments 

        o3d.visualization.draw_geometries([*line_mesh]), #*radius_line_mesh, pcd])  
    return edges, radius, new_scores

def _extract_nurbs(pcd, cluster_indices, degree=1, visualize=False):
    curves = []
    for cluster_idx in range(max(cluster_indices)+1):
        cluster_by_index = np.argwhere(np.array(cluster_indices)==cluster_idx)
        if len(cluster_by_index)==0 or len(cluster_by_index)<50:
            continue
        cluster_pcd = pcd.select_by_index(cluster_by_index)
        cluster_array = np.asarray(cluster_pcd.points)
        cluster_centered = cluster_array - np.mean(cluster_array, axis=0)
        covariance_matrix = (cluster_centered.T@cluster_centered)/len(cluster_centered)
        u, s, vh = np.linalg.svd(covariance_matrix)
        cluster_centered_aligned = cluster_centered@u
        cluster_centered_aligned_sorted = cluster_centered_aligned[cluster_centered_aligned[:, 0].argsort()]
        cluster_centered_sorted = cluster_centered_aligned_sorted@vh
        cluster_sorted = cluster_centered_sorted + np.mean(cluster_array, axis=0)
        curve = fitting.approximate_curve(cluster_sorted.tolist(), degree=degree, ctrlpts_size=3)
        curve_points = np.array(curve.evalpts)
        curves.append(curve_points)
    curves = np.asarray(curves)
    if visualize:
        line_meshes = []
        for curve in curves:
            edge_lines = np.array([[i, i+1] for i in range(len(curve)-1)])
            edge_colors = [[1, 0, 0] for i in range(len(edge_lines))]
            line_mesh = LineMesh(curve, edge_lines, edge_colors, radius=0.001).cylinder_segments
            line_meshes+=[*line_mesh]
        line_meshes.append(pcd)
        o3d.visualization.draw_geometries(line_meshes)
        
    return curves


def refine_cluster(pcd, cluster_indices, scores, dbscan_eps=0.01):
    new_pcd = o3d.geometry.PointCloud() 
    new_cluster_indices = []
    new_scores = []  
    cluster_i = 0  
    for cluster_idx in range(max(cluster_indices)+1):
        cluster_by_index = np.argwhere(np.array(cluster_indices)==cluster_idx)
        if len(cluster_by_index)==0:
            continue
        cluster_pcd = pcd.select_by_index(cluster_by_index)
        labels = np.array(cluster_pcd.cluster_dbscan(eps=dbscan_eps, min_points=3, print_progress=False))
        num_clusters = labels.max()+1
        if num_clusters>1:
            for sub_cluster_idx in range(max(labels)+1):
                sub_cluster_by_index = np.argwhere(np.array(labels)==sub_cluster_idx)
                sub_cluster_pcd = cluster_pcd.select_by_index(sub_cluster_by_index)
                new_pcd += sub_cluster_pcd
                new_cluster_indices = new_cluster_indices + [cluster_i]*len(sub_cluster_by_index)
                new_scores.append(scores[cluster_idx])
                cluster_i+=1
        else:
            new_pcd += cluster_pcd
            new_cluster_indices = new_cluster_indices + [cluster_i]*len(cluster_by_index)
            new_scores.append(scores[cluster_idx])
            cluster_i+=1
    new_scores = np.array(new_scores)
    return new_pcd, new_cluster_indices, new_scores

def extract_edges(pcd, cluster_indices, min_cluster_size=100, visualize=False):
    edges = []
    for cluster_idx in range(max(cluster_indices)+1):
        cluster_by_index = np.argwhere(np.array(cluster_indices)==cluster_idx)
        if len(cluster_by_index)==0 or len(cluster_by_index)<min_cluster_size:
            continue
        cluster_pcd = pcd.select_by_index(cluster_by_index)
        cluster_array = np.asarray(cluster_pcd.points)
        cluster_centered = cluster_array - np.mean(cluster_array, axis=0)
        covariance_matrix = (cluster_centered.T@cluster_centered)/len(cluster_centered)
        u, s, vh = np.linalg.svd(covariance_matrix)
        AR = np.sqrt(s[1]/s[0])
        if AR<0.3:
            cluster_centered_aligned = cluster_centered@u
            x_max = np.max(cluster_centered_aligned[:,0])
            x_min = np.min(cluster_centered_aligned[:,0])
            edge_centered_aligned = np.array([[x_max, 0, 0], [x_min, 0, 0]])
        else:
            edge_centered_aligned = np.array([[0, 0, 0], [0, 0, 0]])
        edge_centered = edge_centered_aligned@vh
        edge = edge_centered + np.mean(cluster_array, axis=0)
        edges.append(edge)
    edges = np.asarray(edges)
    if visualize:
        edge_points = edges.reshape(-1,3)
        edge_lines = np.array([[i,i+1] for i in range(0, len(edge_points), 2)])
        edge_colors = [[1, 0, 0] for i in range(len(edge_lines))]
        line_mesh = LineMesh(edge_points, edge_lines, edge_colors, radius=0.001).cylinder_segments 
        o3d.visualization.draw_geometries([*line_mesh, pcd])
    
    return edges

def estimate_cluster_radius(pcd, cluster_indices, min_cluster_size=100, visualize=False):
    edges = []
    radius = []
    for cluster_idx in range(max(cluster_indices)+1):
        cluster_by_index = np.argwhere(np.array(cluster_indices)==cluster_idx)
        if len(cluster_by_index)==0 or len(cluster_by_index)<min_cluster_size:
            continue
        cluster_pcd = pcd.select_by_index(cluster_by_index)
        cluster_array = np.asarray(cluster_pcd.points)
        cluster_centered = cluster_array - np.mean(cluster_array, axis=0)
        covariance_matrix = (cluster_centered.T@cluster_centered)/len(cluster_centered)
        u, s, vh = np.linalg.svd(covariance_matrix)

        cluster_centered_aligned = cluster_centered@u
        
        mask = cluster_centered_aligned[:,0] < 0.005 # 0.002
        cluster_centered_aligned = cluster_centered_aligned[mask]
        
        mask = cluster_centered_aligned[:,0] > -0.005 # -0.002
        cluster_centered_aligned = cluster_centered_aligned[mask]
        
        y_max = np.max(cluster_centered_aligned[:,1])
        y_min = np.min(cluster_centered_aligned[:,1])
        edge_centered_aligned = np.array([[0, y_max, 0],
                         [0, y_min, 0]])
        edge_centered = edge_centered_aligned@vh
        edge = edge_centered + np.mean(cluster_array, axis=0)
        edges.append(edge)
        radius.append((y_max-y_min)/2)
    edges = np.asarray(edges)
    radius = np.asarray(radius)
    if visualize:
        edge_points = edges.reshape(-1,3)
        edge_lines = np.array([[i,i+1] for i in range(0, len(edge_points), 2)])
        edge_colors = [[1, 0, 0] for i in range(len(edge_lines))]
        line_mesh = LineMesh(edge_points, edge_lines, edge_colors, radius=0.001).cylinder_segments 
        o3d.visualization.draw_geometries([*line_mesh, pcd])
    return radius