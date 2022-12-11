import copy
import distinctipy
import numpy as np
import open3d as o3d
import scipy
import matplotlib.pyplot as plt
import networkx as nx

try:
    from modules.helper.visualization import LineMesh
except ModuleNotFoundError:
    from process_2021.tree_skeletonization.modules.helper.visualization import LineMesh

np.set_printoptions(precision=4, suppress=True)
INF = float("inf")


def make_cloud(points, colors=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def skeletonize(method, edges, radius, likelihood_values, likelihood_points,
                likelihood_colors, voxel_size, clean=False, viz_dir=None):
    '''
    Arguments:
        method: String, choice of ["default", "field", "mst", "ftsem"]
            default: TODO
            field: TODO
            mst: TODO
            ftsem: TODO
        edges: TODO
        radius: TODO
        likelihood_values: TODO
        likelihood_points: TODO
        likelihood_colors: TODO
        voxel_size: TODO
        clean: TODO

    '''

    # These are only produced with certain methods, so make the default None
    map_pcd = None
    tree_mesh = None

    if method in ["default", "field"]:
        map_pcd = make_cloud(likelihood_points, likelihood_colors)
        map_vals_rgb = np.zeros_like(np.array(map_pcd.colors))
        map_vals_rgb[:, 0] = likelihood_values
        map_pcd.colors = o3d.utility.Vector3dVector(map_vals_rgb)
        for voxel_size in np.arange(max(voxel_size, 0.01), 0.1, 0.001):
            map_vg = o3d.geometry.VoxelGrid.create_from_point_cloud(map_pcd, voxel_size=voxel_size)
            num_voxels = len(map_vg.get_voxels())
            # Limited by memory usage.
            # if num_voxels < 55000:
            # TODO: Look into more efficient storage mechanisms that don't need
            # this much space (adjacency matrix inefficient)
            if num_voxels < 25000:
                voxel_size = round(voxel_size, 3)
                print(f"Selected Voxel Size {voxel_size} with {num_voxels} likelihood nodes.")
                break
        else:
            raise RuntimeError("Never found voxel size that met memory requirements")

        map_points = np.asarray([
            map_vg.origin + (pt.grid_index + 0.5) * map_vg.voxel_size
            for pt in map_vg.get_voxels()
        ])
        map_values = np.asarray([pt.color[0] for pt in map_vg.get_voxels()])
        map_pcd = make_cloud(map_points, plt.get_cmap("jet")(map_values)[:,:3])
        # Make a graph on the likelihood voxels that is fully connected within
        # the voxel size radius
        map_tree = UndirectedGraph(map_pcd)
        map_tree.construct_skeleton_graph(voxel_size)
        # map_tree.save_viz(viz_dir, "_LIKELIHOODMAP", linecolor=(0, 0, 1))  # REMOVE

        def fn_weight(u, v, d):
            '''
            u and v are two points in the likelihood map
            We turn likelihood (high good) into cost (low good) with -log
            d is unused, it must be accepted by the function we're calling.
            '''
            return -np.log((map_values[u] + map_values[v]) / 2)

        if clean:
            observed_tree = construct_initial_clean_skeleton(edges, radius)
        else:
            observed_tree = construct_initial_skeleton(edges, radius)
        # observed_tree.save_viz(viz_dir, "_PRESKEL")  # REMOVE
        merger = SkeletonMerger(observed_tree, map_tree, fn_weight)
        main_tree = merger.main_tree

        main_tree.pcd = laplacian_smoothing(main_tree.pcd, search_radius=0.015)
        main_tree.nodes_array = np.array(main_tree.pcd.points)
        #main_tree.laplacian_smoothing()

        main_tree_pcd, radius = main_tree.distribute_equally(0.001) #0.0005 # update radius
        tree_mesh = generate_sphere_mesh(main_tree_pcd, radius) # update_radius
        main_tree.save_viz(viz_dir, "_MERGE")  # REMOVE

    elif method == "mst":
        if clean:
            main_tree = construct_initial_clean_skeleton(edges, radius)
        else:
            observed_tree = construct_initial_skeleton(edges, radius)
            main_tree = UndirectedGraph(observed_tree.distribute_equally(0.01)[0], search_radius_scale=2)
            main_tree.construct_initial_graphs()
            main_tree.merge_components()
        # main_tree.save_viz(viz_dir, "_preMST")  # REMOVE
        main_tree.minimum_spanning_tree()
        main_tree.save_viz(viz_dir, "_postMST")  # REMOVE
        main_tree.pcd = laplacian_smoothing(main_tree.pcd, search_radius=0.015)
        main_tree.nodes_array = np.array(main_tree.pcd.points)
        main_tree.num_nodes = len(main_tree.nodes_array)
        main_tree.laplacian_smoothing()
        main_tree.save_viz(viz_dir, "_postLaplace")  # REMOVE
        main_tree_pcd = main_tree.distribute_equally(0.001)[0]
        main_tree.save_viz(viz_dir, "_postDistribute", main_tree_pcd)  # REMOVE

    elif method == "ftsem":
        if clean:
            main_tree = construct_initial_clean_skeleton(edges, radius)
        else:
            observed_tree = construct_initial_skeleton(edges, radius)
            main_tree = UndirectedGraph(observed_tree.distribute_equally(0.01)[0], search_radius_scale=2)
            main_tree.construct_initial_graphs()
            main_tree.merge_components()
        # main_tree.save_viz(viz_dir, "_preFTSEM")  # REMOVE
        connected = True
        connection_count = 0
        while connected:
            connected = main_tree.breakpoint_connection()
            connection_count += 1
            print('{} breakpoints connected.'.format(connection_count))
        # main_tree.save_viz(viz_dir, "_postFTSEM")  # REMOVE
        main_tree.laplacian_smoothing()
        # main_tree.save_viz(viz_dir, "_postLaplace")  # REMOVE
        main_tree_pcd = main_tree.distribute_equally(0.001)[0]
        # main_tree.save_viz(viz_dir, "_postDistribute", main_tree_pcd)  # REMOVE

    else:
        raise ValueError(f"Found unexpected method {method}")

    return main_tree_pcd, map_pcd, tree_mesh, main_tree.toarray()


def laplacian_smoothing(pcd, search_radius=0.01):
    '''
    Dynamically computes a search radius for each point, then averaging
    the points within the search radius.
    Take the closest 20 points, compute the shape. If the shape is slender
    (first principle bigger than second) then the search radius we use is
    smaller.
    This filters down points by removing points that are in the same spot
    after this averaging process.
    '''
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    filtered_points = []
    for point in np.asarray(pcd.points):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point, knn=20)
        points_in_radius = np.asarray(pcd.points)[idx, :]

        points_in_radius_centered = points_in_radius - np.mean(points_in_radius, axis=0)
        covariance_matrix = (points_in_radius_centered.T@points_in_radius_centered)/len(points_in_radius_centered)
        u, s, vh = np.linalg.svd(covariance_matrix)
        AR = np.sqrt(s[1]/s[0])

        [k, idx, _] = pcd_tree.search_radius_vector_3d(point, radius=0.001+search_radius*AR)
        points_in_radius = np.asarray(pcd.points)[idx, :]
        avg_point = np.mean(points_in_radius, axis=0)
        filtered_points.append(avg_point)
    # filtered_points = np.unique(filtered_points, axis=0)  # BREAKS LAPLACIAN SMOOTHING
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(np.array(filtered_points))
    return filtered_pcd


def equal_spacing(pcd, radiuses): # update radius
    '''
    Basic idea is to diffuse points away from dense areas along the vector made
    by their neighbors.
    If a point is between two neighbors then make it the average, otherwise
    leave it
    '''

    unique_points, unique_index = np.unique(np.array(pcd.points), return_index=True, axis=0)
    radiuses = radiuses[unique_index] # update radius
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(unique_points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    num_points = len(np.asarray(pcd.points))
    total_dist = 0
    for point in np.asarray(pcd.points):
        [k, idx, dist] = pcd_tree.search_knn_vector_3d(point, 2)
        total_dist+=np.sqrt(dist[1])
    radius = total_dist / num_points * 10

    equally_spaced_points = []
    for point in np.asarray(pcd.points):
        [k, idx, dist] = pcd_tree.search_knn_vector_3d(point, 3)
        neighbor_points = np.asarray(pcd.points)[idx[1:], :]
        vec1 = neighbor_points[0] - point
        vec2 = neighbor_points[1] - point
        vec1_length = np.linalg.norm(vec1)
        vec2_length = np.linalg.norm(vec2)
        if vec1_length>0 and vec2_length>0:
            vec1_normalized = vec1/vec1_length
            vec2_normalized = vec2/vec2_length
            cosine_similarity = np.dot(vec1_normalized, vec2_normalized)
        else:
            cosine_similarity = 1
        if cosine_similarity<0 and max(vec1_length, vec2_length)<radius:
            new_point = np.sum(neighbor_points, axis=0)/2
            equally_spaced_points.append(new_point)
        else:
            equally_spaced_points.append(point)

    equally_spaced_pcd = o3d.geometry.PointCloud()
    equally_spaced_pcd.points = o3d.utility.Vector3dVector(np.array(equally_spaced_points))
    equally_spaced_pcd_color = np.zeros_like(equally_spaced_points)
    equally_spaced_pcd_color[:, 1] = 1
    equally_spaced_pcd.colors = o3d.utility.Vector3dVector(equally_spaced_pcd_color)
    return equally_spaced_pcd, radiuses # update radius


class UndirectedGraph:
    def __init__(self, pcd, radius=None, search_radius_scale=10, graph=None):
        '''
        Arguments:
            pcd: TODO
            radius: TODO
            search_radius_scale: TODO
            graph: If not None, then this should be a networkx.Graph object. It
                will be used to set the adjacency_matrix
        '''
        self.pcd = pcd
        self.nodes_radius = np.zeros(len(pcd.points))
        if radius is not None:
            self.nodes_radius = np.array(radius) # update
        self.nodes_array = np.array(pcd.points)
        self.kdtree = o3d.geometry.KDTreeFlann(pcd)
        self.num_nodes = len(self.nodes_array)
        if graph is None:
            self._adjacency_matrix = np.zeros((self.num_nodes, self.num_nodes))
            self._graph = None
        else:
            # TODO: Maybe in the future convert adjacency_matrix to sparse?
            self._adjacency_matrix = nx.adjacency_matrix(graph).toarray()
            self._graph = graph

        # The search radius is the average neighbor-to-neighbor distance scaled
        total_dist = 0
        for node in self.nodes_array:
            # By getting two neighbors we get the node itself and the nearest
            [k, idx, dist] = self.kdtree.search_knn_vector_3d(node, 2)
            # Therefore dist[1] is the distance (squared) to the neighbor
            total_dist += np.sqrt(dist[1])
        self.search_radius = total_dist / self.num_nodes * search_radius_scale

    @property
    def graph(self):
        '''
        In conjunction with set_adjacency, tries to maintain parity between
        adjacency matrix and a corresponding graph
        '''
        if self._graph is None:
            self._graph = nx.from_numpy_matrix(self._adjacency_matrix)
        return self._graph

    @property
    def adjacency_matrix(self):
        '''
        In conjunction with set_adjacency, tries to maintain parity between
        adjacency matrix and a corresponding graph
        '''
        if self._adjacency_matrix is None:
            self._adjacency_matrix = nx.adjacency_matrix(self._graph).toarray()
        return self._adjacency_matrix

    def add_graph_edge(self, idx0, idx1, weight):
        '''
        Sets values to the graph and in the meantime makes sure to sunset the
        adjacency matrix.
        STEPPING STONE - This could be improved but should be robust.
        '''
        self._graph.add_edge(idx0, idx1, weight=weight)
        self._adjacency_matrix = None

    def set_adjacency(self, idx0, idx1, value):
        '''
        Sets values to the adjacency matrix (symmetrically for undirected
        graph) and in the meantime makes sure to sunset the stored graph.
        STEPPING STONE - This could be improved but should be robust.
        '''
        self._adjacency_matrix[idx0, idx1] = value
        self._adjacency_matrix[idx1, idx0] = value
        self._graph = None

    def set_adjacency_matrix(self, matrix):
        '''
        Sets values to the adjacency matrix and in the meantime makes sure to
        sunset the stored graph.
        STEPPING STONE - This could be improved but should be robust.
        '''
        self._adjacency_matrix = matrix
        self._graph = None

    def toarray(self):
        graph = self.graph
        points = self.nodes_array
        radii = self.nodes_radius
        # Format is 8-vector of [3d point 1, 3d point 2, radius 1, radius 2],
        # each of which stores an edge in the skeleton
        return np.array([
            points[e[0]].tolist() + points[e[1]].tolist() + [radii[e[0]], radii[e[1]]]
            for e in graph.edges
            if np.linalg.norm(points[e[1]] - points[e[0]]) > 1e-6
        ])

    def construct_initial_graphs(self):
        '''
        Fills in self.adjacency_matrix by connected nodes up to their nearest
        neighbor, up to a search radius.
        '''
        self.visited_nodes = set()
        for loop_current_node_idx, current_node in enumerate(self.nodes_array):
            self._construct_initial_graph_recursion(loop_current_node_idx)

    def _construct_initial_graph_recursion(self, current_node_idx):
        if current_node_idx in self.visited_nodes:
            return
        self.visited_nodes.add(current_node_idx)
        [k, idx, dist_sq] = self.kdtree.search_radius_vector_3d(
            self.nodes_array[current_node_idx],
            self.search_radius,
        )
        for neighbor_node_idx in idx:
            if neighbor_node_idx not in self.visited_nodes:
                self.set_adjacency(current_node_idx, neighbor_node_idx, 1)
                self._construct_initial_graph_recursion(neighbor_node_idx)
                break
        return

    def construct_skeleton_graph(self, voxel_size):
        '''
        Populate the adjacency matrix by searching through the node array and
        fully connecting all nodes that are within a little greater radius than
        the voxel size.
        '''
        for current_node_idx, current_node in enumerate(self.nodes_array):
            [k, neighbor_idx, dist_sq] = self.kdtree.search_radius_vector_3d(
                self.nodes_array[current_node_idx],
                voxel_size*np.sqrt(3) * 1.1,
            )
            for neighbor_node_idx, dist in zip(neighbor_idx[1:], np.sqrt(dist_sq[1:])):
                self.set_adjacency(current_node_idx, neighbor_node_idx, dist)

    def get_connected_components(self):
        return scipy.sparse.csgraph.connected_components(self.adjacency_matrix)

    def save_viz(self, viz_dir, suffix="", pcd=None, linecolor=(1, 0, 0)):
        if pcd is not None:
            o3d.io.write_point_cloud(str(viz_dir.joinpath(f"finalpcd{suffix}.ply")), pcd)
        combined_pcd = self.visualize_adjacency_matrix(display=False)
        o3d.io.write_point_cloud(str(viz_dir.joinpath(f"adj_components{suffix}.ply")), combined_pcd)
        line_mesh, _ = self.visualize_tree(display=False, linecolor=linecolor)
        o3d.io.write_triangle_mesh(str(viz_dir.joinpath(f"tree_mesh{suffix}.ply")), line_mesh)

    def visualize_adjacency_matrix(self, pcd=None, display=True):
        num_components, components = self.get_connected_components()
        combined_pcd = o3d.geometry.PointCloud()
        for i, color in enumerate(distinctipy.get_colors(num_components)):
            component = self.nodes_array[np.where(components == i)[0]]
            combined_pcd += make_cloud(points=component,
                                       colors=np.ones_like(component)*color)
        if display:
            clouds = [combined_pcd]
            if pcd is not None:
                clouds.append(pcd)
            o3d.visualization.draw_geometries(clouds)
        return combined_pcd

    def merge_components(self):
        '''
        Searches through connected components for nearest neighbors that are
        closer than the search radius (a little bigger than the voxel size),
        and connect those neighbors.

        At the end there will may still be multiple connected components but
        they will be separated by the search radius.
        '''

        no_change = False
        while not no_change:
            num_components, connected_components = self.get_connected_components()
            if num_components == 1:
                break
            print('Merging components. Components remaining:', num_components)
            merged = False
            for i in range(num_components):
                component_node_indices = np.where(connected_components == i)[0]
                other_node_indices = np.where(connected_components != i)[0]
                other_nodes = self.nodes_array[other_node_indices]
                other_pcd = o3d.geometry.PointCloud()
                other_pcd.points = o3d.utility.Vector3dVector(other_nodes)
                other_tree = o3d.geometry.KDTreeFlann(other_pcd)
                min_dist = INF
                for node_idx in component_node_indices:
                    node = self.nodes_array[node_idx]
                    [k, idx, dist_sq] = other_tree.search_knn_vector_3d(node, 1)
                    dist = np.sqrt(dist_sq[0])
                    if dist < min_dist:
                        closest_other_idx = other_node_indices[idx[0]]
                        closest_component_idx = node_idx
                        min_dist = dist
                if min_dist < self.search_radius:
                    assert(self.adjacency_matrix[closest_component_idx, closest_other_idx] != 1)
                    self.set_adjacency(closest_component_idx, closest_other_idx, 1)
                    merged = True
                    break
            if not merged:
                no_change = True

    def bridge_components(self, new_nodes, head_idx, tail_idx):
        '''
        Update the adjacency matrix.
        '''

        new_nodes = np.array(new_nodes)
        num_new_nodes = len(new_nodes)
        new_node_indices = np.array(range(num_new_nodes)) + self.num_nodes

        # update radius
        head_radius = self.nodes_radius[head_idx]
        tail_radius = self.nodes_radius[tail_idx]
        radius_step = (head_radius - tail_radius) / (num_new_nodes + 1)
        for i in range(num_new_nodes):
            self.nodes_radius = np.append(
                self.nodes_radius,
                [head_radius - radius_step * (i + 1)],
            )

        # Update class properties
        self.nodes_array = np.concatenate((self.nodes_array, new_nodes), axis=0)
        self.pcd.points = o3d.utility.Vector3dVector(self.nodes_array)
        self.kdtree = o3d.geometry.KDTreeFlann(self.pcd)
        self.num_nodes = len(self.nodes_array)

        # Add bridge connections
        for idx0, idx1 in zip([head_idx] + new_node_indices.tolist(),
                              new_node_indices.tolist() + [tail_idx]):
            self.add_graph_edge(idx0, idx1, 1)

    def get_edgelist(self):
        nx_graph = nx.from_numpy_matrix(np.triu(self.adjacency_matrix), create_using=nx.DiGraph())
        edge_list = nx.to_edgelist(nx_graph)
        edge_node_list = []
        for edge_idx in edge_list:
            if np.linalg.norm(self.nodes_array[edge_idx[0]]-self.nodes_array[edge_idx[1]])>0.008 * 0.2:
                edge_node = [self.nodes_array[edge_idx[0]], self.nodes_array[edge_idx[1]]]
                edge_node_list.append(edge_node)
        edge_node_array = np.array(edge_node_list)
        return edge_node_array

    def visualize_tree(self, pcd=None, display=True, linecolor=(1, 0, 0)):
        if pcd is None:
            pcd = self.pcd
        edge_points = self.get_edgelist().reshape(-1, 3)
        edge_lines = np.array([[i, i + 1] for i in range(0, len(edge_points), 2)])
        edge_colors = [linecolor] * len(edge_lines)
        line_mesh = o3d.geometry.TriangleMesh()
        for line in LineMesh(edge_points, edge_lines, edge_colors, radius=0.001).cylinder_segments:
            line_mesh += line
        if display:
            o3d.visualization.draw_geometries([line_mesh, pcd])
        return line_mesh, pcd

    def distribute_equally(self, spacing):
        nx_graph = nx.from_numpy_matrix(self.adjacency_matrix, create_using=nx.DiGraph())
        nodes = []
        nodes.append(self.nodes_array[0])
        radius = []
        radius.append(self.nodes_radius[0]) # update radius
        remaining_dist = 0
        for i, edge in enumerate(nx.dfs_edges(nx_graph)):
            e0 = self.nodes_array[edge[0]]
            e1 = self.nodes_array[edge[1]]

            r0 = self.nodes_radius[edge[0]] # update radius
            r1 = self.nodes_radius[edge[1]] # update radius

            edge_dist = np.linalg.norm(e1 - e0)
            if np.isclose(edge_dist, 0):
                unit_vector=None
            else:
                unit_vector = (e1 - e0) / edge_dist
            remaining_dist = edge_dist + remaining_dist
            first_flag = True
            nodes_added = 0 # update radius
            while remaining_dist > spacing:
                if first_flag:
                    new_node = e0 + spacing * unit_vector
                    first_flag = False
                else:
                    new_node = nodes[-1] + spacing * unit_vector
                nodes.append(new_node)
                nodes_added += 1 # update radius
                remaining_dist -= spacing
                if remaining_dist < spacing:
                    break
            if nodes_added == 1:
                r_step = 0
            else:
                r_step = (r0 - r1) / (nodes_added - 1) # update radius
            for i in range(nodes_added): # update radius
                radius.append(r0 - r_step * i) # update radius

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(nodes))
        return pcd, radius # update radius

    def laplacian_smoothing(self):
        '''
        Reposition points so they are in the center of their first neighbors in
        the adjacency matrix.
        '''
        smoothed_nodes_array = np.zeros_like(self.nodes_array)
        total_euclidean_change = INF
        while total_euclidean_change > 0.01:
            total_euclidean_change = 0
            for i in range(self.nodes_array.shape[0]):
                adjacent_nodes = np.argwhere(self.adjacency_matrix[i])
                if len(adjacent_nodes) > 1:
                    new_node_position = np.mean(self.nodes_array[adjacent_nodes], axis=0)
                    smoothed_nodes_array[i] = new_node_position
                    total_euclidean_change += np.linalg.norm(self.nodes_array[i] - new_node_position)
                else:
                    smoothed_nodes_array[i] = self.nodes_array[i]
            self.nodes_array = smoothed_nodes_array
        self.pcd.points = o3d.utility.Vector3dVector(self.nodes_array)

    # Breakpoint method for FTSEM
    def breakpoint_connection(self):

        # 1. Get Main Branch
        connected_components = self.get_connected_components()
        values, counts = np.unique(connected_components[1], return_counts=True)
        values = np.flip(values[np.argsort(counts)])
        min_z = INF
        for i in range(2):
            comp_nodes = self.nodes_array[np.argwhere(connected_components[1]==values[i])].reshape(-1,3)
            z = np.min(comp_nodes[:,2])
            if z < min_z:
                main_branch_comp_id = values[i]
                min_z = z

        # 2. Filter branches: Only keep branches with nodes > P_T
        P_T = 4
        theta_T = 120 # degrees
        B_list = []
        values, counts = np.unique(connected_components[1], return_counts=True)
        for value, count in zip(values, counts):
            if count>P_T and value!=main_branch_comp_id:
                B_list.append(value)
        # 2.1. Sort branch list (B_i) from lowest to highest z
        branch_lowest_points = []
        for branch_comp_idx in B_list:
            branch_nodes = self.nodes_array[np.argwhere(connected_components[1]==branch_comp_idx)].reshape(-1,3)
            branch_lowest_points.append(np.min(branch_nodes[:,2]))
        B_list = np.array(B_list)[np.argsort(branch_lowest_points)]

        # Make mainbranch kd-tree
        main_branch_node_indices = np.argwhere(connected_components[1]==main_branch_comp_id)
        main_branch_coordinates = self.nodes_array[main_branch_node_indices].reshape(-1,3)
        main_branch_min_coordinate = main_branch_coordinates[np.argmin(main_branch_coordinates[:,2])]
        main_branch_pcd = o3d.geometry.PointCloud()
        main_branch_pcd.points = o3d.utility.Vector3dVector(main_branch_coordinates)
        main_branch_kdtree = o3d.geometry.KDTreeFlann(main_branch_pcd)

        # Main Loop
        # TODO: This line is returning [] on grapevines
        main_branch_breakpoint_indices = self.get_breakpoints(main_branch_comp_id, connected_components)
        main_branch_breakpoint_coordinates = self.nodes_array[main_branch_breakpoint_indices]

        for branch_comp_id in B_list:
            breakpoint_indicies = self.get_breakpoints(branch_comp_id, connected_components)
            break_point_coordinates = self.nodes_array[breakpoint_indicies]
            breakpoint_indicies = breakpoint_indicies[np.argsort(np.linalg.norm(break_point_coordinates-main_branch_min_coordinate, axis=1))]
            for P_j in breakpoint_indicies: # For all breakpoints P_j in B_i
                P_j_coordinate = self.nodes_array[P_j]
                # Step 2: q_k <- Find 5 closest breakpoints in mainbranch
                q_k_node_indices = main_branch_breakpoint_indices[np.argsort(np.linalg.norm(main_branch_breakpoint_coordinates-P_j_coordinate, axis=1))]
                if len(q_k_node_indices)>5:
                     q_k_node_indices = q_k_node_indices[:5]
                # Step 3: np <- Find closest point in mainbranch from P_j where alpha>theta_T
                [k, idx, _] = main_branch_kdtree.search_knn_vector_3d(P_j_coordinate, knn=50)
                NP = None
                for i in idx:
                    main_branch_idx = main_branch_node_indices[i][0]
                    _, abg_candidates = self.compute_bdabg_angles(P_j, main_branch_idx)
                    if np.any(abg_candidates[:,0]>theta_T):
                        NP = main_branch_idx
                        break

                # Step 4: Calculate bdabg for q_k
                bdabg_k_candidates = []
                for q_k_idx in q_k_node_indices:
                    bd_k, q_k_abg_candidates = self.compute_bdabg_angles(P_j, q_k_idx)
                    bdabg_k_candidates.append((q_k_idx, bd_k, np.cos(np.radians(q_k_abg_candidates))))
                # Step 4.1: If np is found: calculate bda for np
                if NP is not None:
                    bd_np, np_abg_candidates = self.compute_bdabg_angles(P_j, NP)
                    bda_np_candidates = (NP, bd_np, np.cos(np.radians(np_abg_candidates)))

                # Step 5: 
                valid_q_k = []
                for q_k_node_idx, bd_k, abg_k_list in bdabg_k_candidates:
                    for abg_k in abg_k_list:
                        if (abg_k[0]<=np.cos(np.radians(theta_T))) and (abg_k[1]<=np.cos(np.radians(theta_T))) and (abg_k[2]>=np.abs(np.cos(np.radians(theta_T)))):
                            valid_q_k.append((q_k_node_idx, bd_k, abg_k))
                # If all q_k fail on equation 5 and do not find np, P_j will not be connected
                if len(valid_q_k)==0 and (NP is None):
                    continue
                elif len(valid_q_k)==0 and (NP is not None):
                    # Connect P_j to NP
                    self.set_adjacency(P_j, NP, 1)
                    return True

                # Step 6:
                q_k_min_list = []
                bd_k_min = INF
                for q_k, bd_k, abg_k in valid_q_k:
                    if np.round(bd_k, 4) <= np.round(bd_k_min, 4):
                        bd_k_min = bd_k
                        q_k_min_list.append((q_k, bd_k, abg_k))
                #assert(len(q_k_min_list)==1)
                q_k_min = q_k_min_list[0]
                if NP is None:
                    # Connect P_j to q_k_min
                    self.set_adjacency(P_j, q_k_min[0], 1)
                    return True
                condition_1 = q_k_min[1]<=3*bda_np_candidates[1]
                condition_2 = q_k_min[2][0]<=np.max(bda_np_candidates[2][:,0])
                equation_6 =  condition_1 and condition_2
                if equation_6:
                    # Connect P_j to q_k_min
                    self.set_adjacency(P_j, q_k_min[0], 1)
                    return True
                else:
                    # Connect P_j to NP
                    self.set_adjacency(P_j, NP, 1)
                    return True

        #self.visualize_tree()
        return False

    def get_parents(self, node_idx):
        first_parent_list = np.argwhere(self.adjacency_matrix[node_idx]).flatten()
        possible_parents = []
        for first_parent in first_parent_list:
            second_parent_list = np.argwhere(self.adjacency_matrix[first_parent]).flatten()
            second_parent_list = second_parent_list[second_parent_list != node_idx]
            for second_parent in second_parent_list:
                possible_parents.append([node_idx, first_parent, second_parent])
        return np.array(possible_parents)

    def compute_bdabg_angles(self, P_j, main_branch_idx):
        bd = np.linalg.norm(self.nodes_array[P_j]-self.nodes_array[main_branch_idx])
        P_j_parent_candidates = self.get_parents(P_j)
        main_branch_parent_candidates = self.get_parents(main_branch_idx)
        abg_candidates = []
        for P_j_parent_candidate in P_j_parent_candidates:
            for main_branch_parent_candidate in main_branch_parent_candidates:
                m_nodes = self.nodes_array[P_j_parent_candidate]
                m_vectors = m_nodes[:2]-m_nodes[1:]
                m = np.mean(m_vectors, axis=0)
                m_normalized = m/np.linalg.norm(m)

                n_nodes = self.nodes_array[main_branch_parent_candidate]
                n_vectors = n_nodes[:2]-n_nodes[1:]
                n = np.mean(n_vectors, axis=0)
                n_normalized = n/np.linalg.norm(n)

                l_nodes = np.concatenate((np.flip(m_nodes, axis=0), n_nodes))      
                #l_nodes = np.concatenate((n_nodes, m_nodes)) 

                l_vectors = l_nodes[:5]-l_nodes[1:]   
                l = np.mean(l_vectors, axis=0) 
                l_normalized = l/np.linalg.norm(l)

                cos_alpha = np.dot(m_normalized,n_normalized)
                cos_beta = np.dot(m_normalized,l_normalized)
                cos_gamma = np.dot(n_normalized,l_normalized)
                alpha_deg = np.degrees(np.arccos(cos_alpha))
                beta_deg = np.degrees(np.arccos(cos_beta))
                gamma_deg = np.degrees(np.arccos(cos_gamma))

                abg_candidates.append([alpha_deg, beta_deg, gamma_deg])
        return bd, np.array(abg_candidates)

    def get_breakpoints(self, component_id, connected_components):
        branch_node_indices = np.argwhere(connected_components[1] == component_id)
        breakpoints_node_indices = []
        for branch_node_idx in branch_node_indices:
            if np.sum(self.adjacency_matrix[branch_node_idx]) == 1:
                breakpoints_node_indices.append(branch_node_idx)
        return np.array(breakpoints_node_indices).flatten()

    # MST
    def minimum_spanning_tree(self):

        print("Building connected graph")
        fully_connected_weighted_adj_mat = np.zeros_like(self.adjacency_matrix)
        for i in range(self.num_nodes - 1):
            for j in range(i + 1, self.num_nodes):
                weight = np.linalg.norm(self.nodes_array[i] - self.nodes_array[j])
                fully_connected_weighted_adj_mat[i, j] = weight
                fully_connected_weighted_adj_mat[j, i] = weight
        G = nx.from_numpy_array(fully_connected_weighted_adj_mat)

        print("Calculating MST")
        T = nx.minimum_spanning_tree(G)

        print("Producing MST matrix")
        mst_adj_mat = np.zeros_like(self.adjacency_matrix)
        for edge in sorted(T.edges(data=True)):
            mst_adj_mat[edge[0],edge[1]] = 1
            mst_adj_mat[edge[1],edge[0]] = 1
        self.set_adjacency_matrix(mst_adj_mat)


class SkeletonMerger:
    def __init__(self, main_tree, likelihood_map, fn_weights, iters=200):
        '''
        Arguments:
            main_tree: UndirectedGraph
            likelihood_map: UndirectedGraph
            fn_weights: Takes in (u, v, d) and outputs cost
        '''
        self.main_tree = main_tree
        self.likelihood_map = likelihood_map
        self.fn_weights = fn_weights
        for i in range(iters):
            print(f"Path search iter: {i+1} / (max) {iters}")
            num_components = self.associate_nodes_to_main()
            if num_components == 1:
                break
            merged = self.merge_shortest_path_components()
            if not merged:
                break

    def associate_nodes_to_main(self):

        # 1. Compute skeleton association to observed tree components
        self.nodes_association = np.full(self.likelihood_map.num_nodes, -1)
        num_components, components = self.main_tree.get_connected_components()
        for map_node_idx, map_node in enumerate(self.likelihood_map.nodes_array):
            k, main_tree_idx, _ = self.main_tree.kdtree.search_radius_vector_3d(
                map_node,
                self.main_tree.search_radius,
            )
            # NOTE: We think this is associating each point in the likelihood
            # map to the connected component ID
            if k > 0:
                self.nodes_association[map_node_idx] = components[main_tree_idx[0]]

        # 2. Get component ID in order of lowest component in to highest
        #    component in euclidean space
        component_min_height_list = []
        for component_id in range(num_components):
            component_indices = np.where(components == component_id)[0]
            component_points_array = self.main_tree.nodes_array[component_indices]
            # Get the point with the lowest Z value
            min_height_in_component = np.min(component_points_array[:,2])
            component_min_height_list.append(min_height_in_component)
        self.component_id_root_order = np.argsort(component_min_height_list)

        return num_components

    def merge_shortest_path_components(self):

        for component_id in self.component_id_root_order:

            # 1. Find which nodes I have to compute paths for
            cluster_nodes = np.where(self.nodes_association == component_id)[0]
            assert len(cluster_nodes) > 0  # REMOVE
            # Likelihood points that have been claimed by some other segment
            valid_targets = np.where(np.logical_and(
                self.nodes_association != component_id,
                self.nodes_association != -1,
            ))[0]

            # 2. Compute path for all nodes and find shortest path. This is
            # exhaustive pathing from the source nodes. The paths variable is a
            # dictionary consisting of
            #   {every G index: path to the closest node in cluster_nodes}
            # Note that the path length for indices already in the cluster will
            # be zero.
            path_lengths, paths = nx.multi_source_dijkstra(
                # This is all nodes, not just valid targets
                G=self.likelihood_map.graph,
                # All likelihood points associated with the current root cluster
                sources=set(cluster_nodes),
                weight=self.fn_weights,
            )

            # For some targets there will be no path, so it won't be available,
            # hence the get() call
            if len(valid_targets) == 0:
                import ipdb; ipdb.set_trace()
            shortest_target = valid_targets[
                np.argmin([path_lengths.get(t, INF) for t in valid_targets])
            ]
            if shortest_target in path_lengths:
                shortest_path_length = path_lengths[shortest_target]
                # List of nodes that steps through the graph
                shortest_path_indices = paths[shortest_target]
            else:
                # TODO: Is there a smart way not to re-do computations?
                print(f"No path found from component {component_id} to any other")
                if component_id == self.component_id_root_order[-1]:
                    print("Ending search, went through all components")
                    return False
                else:
                    continue

            print(f"Shortest path from component {component_id} to any other component:"
                  f" {shortest_path_length:.3f} (cost units)")
            print(f"\tPath indices: {shortest_path_indices}")

            # Find the nodes (points in space) in the likelihood map that start
            # and end the shortest path
            shortest_path_nodes = self.likelihood_map.nodes_array[shortest_path_indices]
            head_node, tail_node = shortest_path_nodes[0], shortest_path_nodes[-1]
            # Find the main tree node indices that are closest to the selected
            # likelihood nodes
            k, head_idx, _ = self.main_tree.kdtree.search_knn_vector_3d(head_node, 1)
            k, tail_idx, _ = self.main_tree.kdtree.search_knn_vector_3d(tail_node, 1)
            head_idx, tail_idx = head_idx[0], tail_idx[0]

            # Bridge from the head to tail (both already part of the main tree)
            # through the shortest path that we found in the likelihoods
            self.main_tree.bridge_components(shortest_path_nodes, head_idx, tail_idx)
            print(f"\tBridged from {self.main_tree.nodes_array[head_idx]} to"
                  f" {self.main_tree.nodes_array[tail_idx]} along the shortest likelihood path")

            return True

    def plot_association(self):
        skeleton_pcd = o3d.geometry.PointCloud()
        skeleton_pcd.points = o3d.utility.Vector3dVector(self.likelihood_map.nodes_array)
        skeleton_pcd_colors = np.zeros_like(skeleton_pcd.points)
        no_association_idx = np.where(self.nodes_association==-1)[0]
        association_idx = np.where(self.nodes_association!=-1)[0]
        skeleton_pcd_colors[no_association_idx] = np.array([1,0,0])
        skeleton_pcd_colors[association_idx] = np.array([0,0,1])
        skeleton_pcd.colors = o3d.utility.Vector3dVector(skeleton_pcd_colors)
        main_pcd = self.main_tree.visualize_adjacency_matrix()
        o3d.visualization.draw_geometries([skeleton_pcd, main_pcd])


def construct_initial_skeleton(edges, radius):
    '''
    First convert edges into a series of linearly spaced points along the
    edges, which are presumed to overlap.
    Does Laplacian smoothing on the overlapped and sampled edges.
    '''

    # Convert edges into points
    edges = np.array(edges)
    edges = np.concatenate(edges)
    edges = edges.reshape(-1,2,3)
    spacing = 0.002 # 0.01 on real tree?
    points = []
    radiuses = [] # update radius
    for edge, r in zip(edges, radius):  # update radius
        vec = edge[1]-edge[0]
        vec_length = np.linalg.norm(vec)
        if vec_length==0:
            points.append(edge[0])
            radiuses.append(r) # update radius
        else:
            vec_normalized = vec/vec_length
            for step in np.arange(0, vec_length, spacing):
                point = edge[0]+vec_normalized*step
                points.append(point)
                radiuses.append(r) # update radius
    radiuses = np.array(radiuses) # update radius
    points_pcd = o3d.geometry.PointCloud()
    points_pcd.points = o3d.utility.Vector3dVector(np.array(points))
    points_color = np.zeros_like(points)
    points_color[:, 2] = 1
    points_pcd.colors = o3d.utility.Vector3dVector(points_color)

    # Apply filtering
    filtered_pcd = copy.deepcopy(points_pcd)
    for i in range(2):
        filtered_pcd = laplacian_smoothing(filtered_pcd, search_radius=0.03)#0.015)

    # Apply equal spacing
    equally_spaced_pcd = copy.deepcopy(filtered_pcd)
    for i in range(10):
        equally_spaced_pcd, radiuses = equal_spacing(equally_spaced_pcd, radiuses) # update radius

    main_tree = UndirectedGraph(equally_spaced_pcd, radiuses) # update radius
    main_tree.construct_initial_graphs()
    main_tree.merge_components()
    return main_tree


def construct_initial_clean_skeleton(edges, radii):
    '''
    Similar to construct_initial_skeleton, but it assumes that the edges are
    already essentially in a graph format, with overlapping points where the
    graph indices should be.

    In contrast, construct_initial_skeleton was designed to be run on edges
    that may be arbitrarily placed and have significant overlapping.

    NOTE: right now all points that match to 6 decimal points are considered
    the same, if that needs to be more flexible it could be changed.

    Arguments:
        edges: Array of shape (M, 2, 3), where M is the number of edges and
            each edge has two associated points. Note that non-leaf points will
            show up in two edges.
        radii: Array of shape (M,), where M is the number of edges. Contains
            the radius of that edge in meters.

    Returns:
        UndirectedGraph object that TODO
    '''

    def pt_to_key(point):
        '''Take a 3D point and make it a hashable key.'''
        return f"({point[0]:.6f},{point[1]:.6f},{point[2]:.6f})"

    # Take our representation of edges/radii and turn them into
    # 1) points
    # 2) radii reindexed by points
    # 3) a networkx undirected graph
    points = []
    reindexed_radii = []
    seen = {}
    graph = nx.Graph()
    for edge, radius in zip(edges, radii):
        for point in edge:
            key = pt_to_key(point)
            if key not in seen:
                seen[key] = len(points)
                points.append(point)
                reindexed_radii.append(radius)
                graph.add_node(seen[key])
        e0, e1 = edge
        graph.add_edge(seen[pt_to_key(e0)],
                       seen[pt_to_key(e1)],
                       weight=np.linalg.norm(e1 - e0))

    return UndirectedGraph(
        pcd=make_cloud(np.array(points)),
        radius=np.array(reindexed_radii),
        graph=graph,
    )


def generate_sphere_mesh(pcd, radius):
    tree_mesh = o3d.geometry.TriangleMesh()
    for point, r in zip(np.array(pcd.points), radius):
        sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=20)
        sphere_mesh.translate(point)
        sphere_mesh.paint_uniform_color([0.588, 0.294, 0])
        tree_mesh+=sphere_mesh
    return tree_mesh