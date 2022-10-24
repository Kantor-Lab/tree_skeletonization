import open3d as o3d
import numpy as np
#from ..skeleton_graph.skeleton_graph import TreeGraph, extract_skeleton

def xyz_spherical(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = np.sqrt(x * x + y * y + z * z)
    r_x = np.arccos(y / r)
    r_y = np.arctan2(z, x)
    return [r, r_x, r_y]

def get_rotation_matrix(r_x, r_y):
    rot_x = np.asarray([[1, 0, 0], [0, np.cos(r_x), -np.sin(r_x)],
                        [0, np.sin(r_x), np.cos(r_x)]])
    rot_y = np.asarray([[np.cos(r_y), 0, np.sin(r_y)], [0, 1, 0],
                        [-np.sin(r_y), 0, np.cos(r_y)]])
    return rot_y.dot(rot_x)

def get_extrinsic(xyz):
    rvec = xyz_spherical(xyz)
    r = get_rotation_matrix(rvec[1], rvec[2])
    t = np.asarray([0, 0, 3]).transpose()
    trans = np.eye(4)
    trans[:3, :3] = r
    trans[:3, 3] = t
    return trans

def preprocess(model):
    min_bound = model.get_min_bound()
    max_bound = model.get_max_bound()
    center = min_bound + (max_bound - min_bound) / 2.0
    #scale = np.linalg.norm(max_bound - min_bound) / 2.0
    vertices = np.asarray(model.vertices)
    vertices -= center
    #model.vertices = o3d.utility.Vector3dVector(vertices / scale)
    model.vertices = o3d.utility.Vector3dVector(vertices)
    return model, center


def voxel_carving(mesh,
                  cubic_size,
                  voxel_resolution,
                  w=1000,
                  h=1000,
                  use_depth=True,
                  surface_method='pointcloud'):
    mesh.compute_vertex_normals()
    #camera_sphere = o3d.io.read_triangle_mesh(camera_path)
    camera_sphere = o3d.geometry.TriangleMesh.create_sphere()

    # setup dense voxel grid
    voxel_carve = o3d.geometry.VoxelGrid.create_dense(
        width=cubic_size,
        height=cubic_size,
        depth=cubic_size,
        voxel_size=cubic_size / voxel_resolution,
        origin=[-cubic_size / 2.0, -cubic_size / 2.0, -cubic_size / 2.0],
        color=[1.0, 0.7, 0.0])

    # rescale geometry
    camera_sphere, _ = preprocess(camera_sphere)
    mesh, offset = preprocess(mesh)

    # setup visualizer to render depthmaps
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h, visible=False)
    vis.add_geometry(mesh)
    vis.get_render_option().mesh_show_back_face = True
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()

    # carve voxel grid
    pcd_agg = o3d.geometry.PointCloud()
    centers_pts = np.zeros((len(camera_sphere.vertices), 3))
    for cid, xyz in enumerate(camera_sphere.vertices):
        # get new camera pose
        trans = get_extrinsic(xyz)
        param.extrinsic = trans
        c = np.linalg.inv(trans).dot(np.asarray([0, 0, 0, 1]).transpose())
        centers_pts[cid, :] = c[:3]
        ctr.convert_from_pinhole_camera_parameters(param)

        # capture depth image and make a point cloud
        vis.poll_events()
        vis.update_renderer()
        depth = vis.capture_depth_float_buffer(False)
        pcd_agg += o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(depth),
            param.intrinsic,
            param.extrinsic,
            depth_scale=1)

        # depth map carving method
        if use_depth:
            voxel_carve.carve_depth_map(o3d.geometry.Image(depth), param)
        else:
            voxel_carve.carve_silhouette(o3d.geometry.Image(depth), param)
        print("Carve view %03d/%03d" % (cid + 1, len(camera_sphere.vertices)))
    vis.destroy_window()

    # add voxel grid survace
    print('Surface voxel grid from %s' % surface_method)
    if surface_method == 'pointcloud':
        voxel_surface = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            pcd_agg,
            voxel_size=cubic_size / voxel_resolution,
            min_bound=(-cubic_size / 2, -cubic_size / 2, -cubic_size / 2),
            max_bound=(cubic_size / 2, cubic_size / 2, cubic_size / 2))
    elif surface_method == 'mesh':
        voxel_surface = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            mesh,
            voxel_size=cubic_size / voxel_resolution,
            min_bound=(-cubic_size / 2, -cubic_size / 2, -cubic_size / 2),
            max_bound=(cubic_size / 2, cubic_size / 2, cubic_size / 2))
    else:
        raise Exception('invalid surface method')
    voxel_carving_surface = voxel_surface + voxel_carve
    return voxel_carving_surface, offset, mesh #voxel_carve, voxel_surface 

def voxelgrid_to_pointcloud(voxel_grid):
    voxel_points = np.asarray([voxel_grid.get_voxel_center_coordinate(pt.grid_index) for pt in voxel_grid.get_voxels()])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxel_points)
    return pcd

if __name__=='__main__':
    mesh_path = 'urdf/speedtree/tree_3/branch/branch.obj'
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([mesh, mesh_frame])

    cubic_size = 1
    voxel_resolution = 200 #400.0
    cam_resolution = 1200 #1200

    voxel_grid, voxel_carve, voxel_surface = voxel_carving(
        mesh, cubic_size, voxel_resolution, cam_resolution, cam_resolution, use_depth=False)


    #print("surface voxels")
    #print(voxel_surface)
    #o3d.visualization.draw_geometries([voxel_surface, mesh_frame])

    #print("carved voxels")
    #print(voxel_carve)
    #o3d.visualization.draw_geometries([voxel_carve, mesh_frame])

    print("combined voxels (carved + surface)")
    print(voxel_grid)
    o3d.visualization.draw_geometries([voxel_grid, mesh_frame])

    pcd = voxelgrid_to_pointcloud(voxel_grid)

    o3d.visualization.draw_geometries([pcd, mesh_frame])

    #skeleton_pcd = extract_skeleton(pcd, voxel_size=0.02)
    #o3d.visualization.draw_geometries([mesh, skeleton_pcd, mesh_frame])

    #tree = TreeGraph(skeleton_pcd)
    #tree.construct_graph(np.asarray(skeleton_pcd.points), 0, voxel_size=0.02) # voxel_size=0.008

    #skeletal_mesh = tree.plot_tree()


