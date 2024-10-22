import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal
from tqdm import tqdm

def construct_likelihood_map(edges, radius, scores, voxel_size, visualize=False): # voxel_size should match that of octomap voxel size
    points_list = []
    likelihood_functions = []
    new_scores = []
    for edge, r, score in zip(edges, radius, scores):
        points, likelihood_fn = edge2likelihood_points(edge, r, voxel_size)    
        if not len(points)==0:
            points_list.append(points)
            likelihood_functions.append(likelihood_fn)
            new_scores.append(score)
    points = np.concatenate(points_list)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd, likelihood = compute_joint_likelihood(pcd, likelihood_functions, new_scores)    

    # Process all points into joint function
    jet_color_map = plt.get_cmap('jet')
    colors = jet_color_map(likelihood)[:,:3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if visualize:
        o3d.visualization.draw_geometries([pcd])
    return pcd, likelihood

def edge2likelihood_points(edge, radius, voxel_size): 
    k = 3
    edge_length = np.linalg.norm(edge[1]-edge[0])
    std1 = (edge_length/3)*k
    std2 = (radius/3)*k
    var1 = std1**2
    var2 = std2**2
    
    if edge_length>0:
        variance = np.array([var1, var2, var2])
        vec1 = (edge[1]-edge[0])/edge_length
        R_mat1 = get_rotation_matrix(vec1)
        # The a,b,c params here need to scale based on how much wide I want to sample
        ellipsoid_samples = sample_ellipsoid(std1*3,std2*3,std2*3, voxel_size) 
    else:
        variance = np.array([var2, var2, var2])
        R_mat1 = np.eye(3)
        ellipsoid_samples = sample_ellipsoid(std2*3,std2*3,std2*3, voxel_size)
    if len(ellipsoid_samples)==0:
        return ellipsoid_samples, None
    
    points, likelihood_fn = gaussian_kernel((edge[1]+edge[0])/2, variance, R_mat1, ellipsoid_samples)
    return points, likelihood_fn 

def get_rotation_matrix(vec2, vec1=np.array([1, 0, 0])):
    """get rotation matrix between two vectors using scipy"""
    vec1 = np.reshape(vec1, (1, -1))
    vec2 = np.reshape(vec2, (1, -1))
    r = Rotation.align_vectors(vec2, vec1)
    return r[0].as_matrix()

def sample_ellipsoid(a,b,c, voxel_size):
    x = np.arange(-a, a+voxel_size, voxel_size)
    y = np.arange(-b, b+voxel_size, voxel_size)
    z = np.arange(-c, c+voxel_size, voxel_size)
    xv, yv, zv= np.meshgrid(x, y, z)
    points = np.concatenate(([xv.flatten()], [yv.flatten()], [zv.flatten()]), axis=0).T
    mask = points[:,0]**2/a**2 + points[:,1]**2/b**2 + points[:,2]**2/c**2<=1
    points = points[mask]
    
    return points

def gaussian_kernel(mean, variance, R_mat, points):
    # Compute covariance matrix
    var = np.diag(variance)
    cov = R_mat@var@R_mat.T
    likelihood_fn = multivariate_normal(cov = cov, mean = mean)    
    points = (R_mat@points.T).T+mean
    return points, likelihood_fn

def compute_joint_likelihood(pcd, likelihood_functions, scores):
    radius_scale = 7 # 99.7% of the points will be within 3 std
    min_hyperparam = 0.0
    pcd_array = np.asarray(pcd.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    likelihoods = np.zeros(pcd_array.shape[0])
    for i in tqdm(range(len(likelihood_functions))):
        fn = likelihood_functions[i]
        eigenvalues, eigenvectors = np.linalg.eig(fn.cov)
        stddev = np.sqrt(eigenvalues).max()
        update_radius = radius_scale*np.real(stddev) # Eig returns X+0j, so we need to take the real part
        # Only update likelihoods within reasonable distance for computational efficiency
        _, indices, dist_sq = kdtree.search_radius_vector_3d(fn.mean, update_radius)
        indices = np.array(indices)
        likelihood = fn.pdf(pcd_array[indices])
        likelihood = (scores[i]-min_hyperparam)*likelihood/max(likelihood)+min_hyperparam 
        likelihoods[indices] = 1 - (1-likelihoods[indices])*(1-likelihood) # Equation 2 in the paper
    return pcd, likelihoods