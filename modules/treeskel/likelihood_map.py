import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal


def construct_likelihood_map(edges, radius, scores, voxel_size,
                             visualize=False, print_freq=1):
    '''
    TODO

    Arguments:
        edges: Array of shape (N, 2, 3) where each edge consists of two 3D
            points
        radius: Array of shape (N,) containing the radius of each edge (m)
        scores: Array of shape (N,) containing the score for each edge
            indicating edge quality. Set to 1 for all edges if relative quality
            is not known.
        voxel_size: TODO
        visualize: Boolean. If true, tries to open an Open3d visualization
            window for the likelihood map-as-pcd
        print_freq: How many iterations of joint likelihoods between printouts.
            Set to a negative number to disable printing

    Returns: Two-element tuple containing
        pcd: Open3d PointCloud object containing a colorized representation of
            the likelihood map (voxel downsampled to voxel_size)
        likelihood: Array of shape (M,), where M is the number of elements in
            pcd.points. This gives the (float, 0-1) likelihood score for each
            point in the space.
    '''
    points_list = []
    likelihood_functions = []
    new_scores = []
    for edge, r, score in zip(edges, radius, scores):
        points, likelihood_fn = edge2likelihood_points(edge, r, voxel_size)
        if not len(points) == 0:
            points_list.append(points)
            likelihood_functions.append(likelihood_fn)
            new_scores.append(score)
    points = np.concatenate(points_list)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd, likelihood = compute_joint_likelihood(pcd,
                                               likelihood_functions,
                                               new_scores,
                                               print_freq)

    # Process all points into joint function
    jet_color_map = plt.get_cmap('jet')
    colors = jet_color_map(likelihood)[:,:3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if visualize:
        o3d.visualization.draw_geometries([pcd])
    return pcd, likelihood


def edge2likelihood_points(edge, radius, voxel_size):
    k = 3
    edge_length = np.linalg.norm(edge[1] - edge[0])
    std1 = (edge_length / 3) * k
    std2 = (radius / 3) * k
    var1 = std1**2
    var2 = std2**2

    if edge_length>0:
        variance = np.array([var1, var2, var2])
        vec1 = (edge[1] - edge[0]) / edge_length
        R_mat1 = get_rotation_matrix(vec1)
        # The a,b,c params need to scale based on how wide I want to sample
        ellipsoid_samples = sample_ellipsoid(std1 * 3,std2 * 3,std2 * 3, voxel_size)
    else:
        variance = np.array([var2, var2, var2])
        R_mat1 = np.eye(3)
        ellipsoid_samples = sample_ellipsoid(std2 * 3,std2 * 3,std2 * 3, voxel_size)

    if len(ellipsoid_samples) == 0:
        return ellipsoid_samples, None

    return gaussian_kernel(
        (edge[1] + edge[0]) / 2,
        variance,
        R_mat1,
        ellipsoid_samples,
    )


def get_rotation_matrix(vec2, vec1=np.array([1, 0, 0])):
    """get rotation matrix between two vectors using scipy"""
    vec1 = np.reshape(vec1, (1, -1))
    vec2 = np.reshape(vec2, (1, -1))
    r = Rotation.align_vectors(vec2, vec1)
    return r[0].as_matrix()


def sample_ellipsoid(a,b,c, voxel_size):
    x = np.arange(-a, a + voxel_size, voxel_size)
    y = np.arange(-b, b + voxel_size, voxel_size)
    z = np.arange(-c, c + voxel_size, voxel_size)
    xv, yv, zv= np.meshgrid(x, y, z)
    points = np.concatenate(([xv.flatten()], [yv.flatten()], [zv.flatten()]), axis=0).T
    mask = (points[:, 0]**2 / a**2 + \
            points[:, 1]**2 / b**2 + \
            points[:, 2]**2 / c**2) <= 1
    points = points[mask]
    return points


def gaussian_kernel(mean, variance, R_mat, points):
    # Compute covariance matrix
    var = np.diag(variance)
    cov = R_mat@var@R_mat.T
    likelihood_fn = multivariate_normal(cov = cov, mean = mean)
    points = (R_mat@points.T).T+mean
    return points, likelihood_fn


def compute_joint_likelihood(pcd, likelihood_functions, scores, print_freq=1):
    min_hyperparam = 0.0
    pcd_array = np.asarray(pcd.points)
    for i, (fn, score) in enumerate(zip(likelihood_functions, scores)):
        if print_freq > 0:
            if i % print_freq == 0:
                print('Computing joint likelihood: {}/{}'.format(i,len(scores)))
        likelihood = fn.pdf(pcd_array)
        likelihood = (score - min_hyperparam) * likelihood / max(likelihood) + min_hyperparam
        if i == 0:
            likelihoods = np.copy(likelihood)
        else:
            likelihoods = 1 - (1 - likelihoods) * (1 - likelihood)
    return pcd, likelihoods