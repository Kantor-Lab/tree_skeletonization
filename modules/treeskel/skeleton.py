import copy
import numpy as np
import open3d as o3d
import scipy.sparse
import networkx as nx
from modules.helper.visualization import LineMesh

class UndirectedGraph:
    def __init__(self, pcd, radius=None, search_radius_scale=10):
        """
        Initialize the UndirectedGraph object.

        Parameters:
        - pcd: open3d.geometry.PointCloud
            The point cloud data.
        - radius: array-like, optional
            The radius of each node.
        - search_radius_scale: float, optional
            The scale factor for search radius.
        """
        self.pcd = pcd
        self.nodes_radius = np.zeros(len(pcd.points))
        if radius is not None:
            self.nodes_radius = np.array(radius)
        self.nodes_array = np.array(pcd.points)
        self.kdtree = o3d.geometry.KDTreeFlann(pcd)
        self.num_nodes = len(self.nodes_array)
        self.adjacency_matrix = scipy.sparse.lil_matrix((self.num_nodes, self.num_nodes))
        total_dist = np.sum([np.sqrt(dist[1]) for _, _, dist in (self.kdtree.search_knn_vector_3d(node, 2) for node in self.nodes_array)])
        self.search_radius = total_dist / self.num_nodes * search_radius_scale
    
    def set_adjacency(self, idx0, idx1, value):
        """
        Sets values in the adjacency matrix symmetrically for an undirected graph.

        Parameters:
        - idx0: int
            Index of the first node.
        - idx1: int
            Index of the second node.
        - value: float
            Value to set in the adjacency matrix.
        """
        self.adjacency_matrix[idx0, idx1] = value
        self.adjacency_matrix[idx1, idx0] = value

    def construct_initial_graphs(self):
        """
        Constructs the initial graph using recursion to connect nodes.
        """
        self.visited_nodes = set()
        for loop_current_node_idx in range(len(self.nodes_array)):
            self._construct_initial_graph_recursion(loop_current_node_idx)
     
    def _construct_initial_graph_recursion(self, current_node_idx):
        """
        Recursively constructs the initial graph by connecting nodes within the search radius.

        Parameters:
        - current_node_idx: int
            Index of the current node.
        """
        if current_node_idx in self.visited_nodes:
            return 
        self.visited_nodes.add(current_node_idx)            
        _, idx, _ = self.kdtree.search_radius_vector_3d(self.nodes_array[current_node_idx], self.search_radius)
        for neighbor_node_idx in idx:
            if neighbor_node_idx not in self.visited_nodes:
                self.set_adjacency(current_node_idx, neighbor_node_idx, 1)
                self._construct_initial_graph_recursion(neighbor_node_idx)
                break

    def construct_skeleton_graph(self, voxel_size):
        """
        Constructs the skeleton graph by connecting nodes within a scaled voxel size radius.

        Parameters:
        - voxel_size: float
            The size of the voxel.
        """
        for current_node_idx, current_node in enumerate(self.nodes_array):
            _, neighbor_idx, dist_sq = self.kdtree.search_radius_vector_3d(self.nodes_array[current_node_idx], voxel_size * np.sqrt(3) * 1.1)
            for neighbor_node_idx, dist in zip(neighbor_idx[1:], np.sqrt(dist_sq[1:])):
                self.set_adjacency(current_node_idx, neighbor_node_idx, dist)
 
    def get_connected_components(self):
        """
        Gets the connected components of the graph.

        Returns:
        - tuple
            Number of connected components and an array indicating the component assignment of each node.
        """
        return scipy.sparse.csgraph.connected_components(self.adjacency_matrix)  

    def visualize_adjacency_matrix(self, pcd=None):
        """
        Visualizes the adjacency matrix by creating point clouds for each connected component.

        Parameters:
        - pcd: open3d.geometry.PointCloud, optional
            Additional point cloud to visualize.

        Returns:
        - open3d.geometry.PointCloud
            Combined point cloud of all connected components.
        """
        num_components, connected_components = self.get_connected_components()
        combined_pcd = o3d.geometry.PointCloud()
        for i in range(num_components):
            result = np.where(connected_components == i)[0]
            component = self.nodes_array[result]
            component_pcd = o3d.geometry.PointCloud()
            component_pcd.points = o3d.utility.Vector3dVector(component)    
            component_pcd_color = np.zeros_like(component)
            component_pcd_color[:] = np.random.uniform(0, 1, 3)   
            component_pcd.colors = o3d.utility.Vector3dVector(component_pcd_color)
            combined_pcd += component_pcd
        if not pcd:
            o3d.visualization.draw_geometries([combined_pcd])
        else:
            o3d.visualization.draw_geometries([combined_pcd, pcd])
        return combined_pcd

    def merge_components(self):
        """
        Merges the components that are within the radius threshold for the initial skeleton.
        """
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
                other_tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(other_nodes)))
                min_dist = float('inf')
                for node_idx in component_node_indices:
                    _, idx, dist_sq = other_tree.search_knn_vector_3d(self.nodes_array[node_idx], 1)
                    dist = np.sqrt(dist_sq[0])
                    if dist < min_dist:
                        closest_other_idx = other_node_indices[idx[0]]
                        closest_component_idx = node_idx
                        min_dist = dist
                if min_dist < self.search_radius:
                    assert self.adjacency_matrix[closest_component_idx, closest_other_idx] != 1
                    self.set_adjacency(closest_component_idx, closest_other_idx, 1)
                    merged = True
                    break
            if not merged:
                no_change = True

    def bridge_components(self, new_nodes, head_idx, tail_idx):
        """
        Bridges components by adding new nodes and updating the adjacency matrix.

        Parameters:
        - new_nodes: array-like
            New nodes to be added.
        - head_idx: int
            Index of the head node.
        - tail_idx: int
            Index of the tail node.
        """
        new_nodes = np.array(new_nodes)
        num_new_nodes = len(new_nodes)
        new_node_indices = np.arange(self.num_nodes, self.num_nodes + num_new_nodes)

        # update radius
        head_radius = self.nodes_radius[head_idx]
        tail_radius = self.nodes_radius[tail_idx]
        radius_step = (head_radius - tail_radius) / (num_new_nodes + 1)
        new_radii = head_radius - radius_step * np.arange(1, num_new_nodes + 1)
        self.nodes_radius = np.append(self.nodes_radius, new_radii)

        # Update class properties
        self.nodes_array = np.vstack((self.nodes_array, new_nodes))
        self.pcd.points = o3d.utility.Vector3dVector(self.nodes_array)    
        self.kdtree = o3d.geometry.KDTreeFlann(self.pcd)
        self.num_nodes = len(self.nodes_array)
        
        new_adjacency_matrix_size = self.adjacency_matrix.shape[0] + num_new_nodes
        new_adjacency_matrix = scipy.sparse.lil_matrix((new_adjacency_matrix_size, new_adjacency_matrix_size))
        new_adjacency_matrix[:self.adjacency_matrix.shape[0], :self.adjacency_matrix.shape[1]] = self.adjacency_matrix 
        self.adjacency_matrix = new_adjacency_matrix        

        # Add bridge connections
        self.set_adjacency(head_idx, new_node_indices[0], 1)
        for i in range(num_new_nodes - 1):
            self.set_adjacency(new_node_indices[i], new_node_indices[i + 1], 1)
        self.set_adjacency(new_node_indices[-1], tail_idx, 1)

    def get_edgelist(self):
        """
        Gets the list of edges in the graph.

        Returns:
        - np.ndarray
            Array of edges in the graph.
        """
        nx_graph = nx.from_scipy_sparse_array(self.adjacency_matrix, create_using=nx.DiGraph())
        edge_list = nx.to_edgelist(nx_graph)
        edge_node_list = [[self.nodes_array[edge[0]], self.nodes_array[edge[1]]] for edge in edge_list if np.linalg.norm(self.nodes_array[edge[0]] - self.nodes_array[edge[1]]) > 0.008 * 0.2]
        return np.array(edge_node_list)

    def visualize_tree(self, pcd=None):
        """
        Visualizes the tree structure using LineMesh.

        Parameters:
        - pcd: open3d.geometry.PointCloud, optional
            Additional point cloud to visualize.
        """
        if pcd is None:
            pcd = self.pcd
        edges = self.get_edgelist()
        edge_points = edges.reshape(-1, 3)
        edge_lines = np.array([[i, i + 1] for i in range(0, len(edge_points), 2)])
        edge_colors = [[1, 0, 0] for _ in range(len(edge_lines))]
        line_mesh = LineMesh(edge_points, edge_lines, edge_colors, radius=0.001).cylinder_segments 
        o3d.visualization.draw_geometries([*line_mesh, pcd])  # self.pcd

    def distribute_equally(self, spacing):
        """
        Distributes nodes equally along the graph edges.

        Parameters:
        - spacing: float
            Desired spacing between nodes.

        Returns:
        - tuple
            Point cloud with equally spaced nodes and their corresponding radii.
        """
        nx_graph = nx.from_scipy_sparse_array(self.adjacency_matrix, create_using=nx.DiGraph())
        dfs = list(nx.dfs_edges(nx_graph))
        nodes = [self.nodes_array[0]]
        radius = [self.nodes_radius[0]]
        remaining_dist = 0
        for edge in dfs:
            e0 = self.nodes_array[edge[0]]
            e1 = self.nodes_array[edge[1]]
            r0 = self.nodes_radius[edge[0]]
            r1 = self.nodes_radius[edge[1]]
            edge_dist = np.linalg.norm(e1 - e0)
            unit_vector = (e1 - e0) / edge_dist if edge_dist != 0 else None
            remaining_dist += edge_dist
            nodes_added = 0
            while remaining_dist > spacing:
                new_node = nodes[-1] + spacing * unit_vector if nodes_added > 0 else e0 + spacing * unit_vector
                nodes.append(new_node)
                nodes_added += 1
                remaining_dist -= spacing
            r_step = (r0 - r1) / max(1, nodes_added - 1)
            radius.extend([r0 - r_step * i for i in range(nodes_added)])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(nodes))
        return pcd, np.array(radius)

    def graph_laplacian_smoothing(self):
        """
        Applies Laplacian smoothing to the graph by averaging node positions with their adjacent nodes.
        """
        smoothed_nodes_array = np.zeros_like(self.nodes_array)
        total_eucledian_change = float('inf')
        while total_eucledian_change > 0.01:
            total_eucledian_change = 0
            for i in range(self.num_nodes):
                adjacent_nodes = self.adjacency_matrix[i].nonzero()[1]
                if len(adjacent_nodes) > 1:
                    new_node_position = np.mean(self.nodes_array[adjacent_nodes], axis=0)
                    smoothed_nodes_array[i] = new_node_position
                    total_eucledian_change += np.linalg.norm(self.nodes_array[i] - new_node_position)
                else:
                    smoothed_nodes_array[i] = self.nodes_array[i]
            self.nodes_array = smoothed_nodes_array
        self.pcd.points = o3d.utility.Vector3dVector(self.nodes_array)

    def breakpoint_connection(self):
        """
        Implements the breakpoint connection method for FTSEM.

        Returns:
        - bool
            True if a connection is made, False otherwise.
        """
        connected_components = self.get_connected_components()
        values, counts = np.unique(connected_components[1], return_counts=True)
        values = np.flip(values[np.argsort(counts)])
        min_z = float('inf')
        for i in range(2):
            comp_nodes = self.nodes_array[np.argwhere(connected_components[1] == values[i])].reshape(-1, 3)
            z = np.min(comp_nodes[:, 2])
            if z < min_z:
                main_branch_comp_id = values[i]
                min_z = z
        
        P_T = 4
        theta_T = 120  # degrees
        B_list = [value for value, count in zip(values, counts) if count > P_T and value != main_branch_comp_id]
        branch_lowest_points = [np.min(self.nodes_array[np.argwhere(connected_components[1] == branch_comp_idx)][:, 2]) for branch_comp_idx in B_list]
        B_list = np.array(B_list)[np.argsort(branch_lowest_points)]

        main_branch_node_indices = np.argwhere(connected_components[1] == main_branch_comp_id).flatten()
        main_branch_coordinates = self.nodes_array[main_branch_node_indices]
        main_branch_min_coordinate = main_branch_coordinates[np.argmin(main_branch_coordinates[:, 2])]
        main_branch_kdtree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(main_branch_coordinates)))

        main_branch_breakpoint_indicies = self.get_breakpoints(main_branch_comp_id, connected_components)
        main_branch_breakpoint_coordinates = self.nodes_array[main_branch_breakpoint_indicies]

        for branch_comp_id in B_list:
            breakpoint_indicies = self.get_breakpoints(branch_comp_id, connected_components)
            break_point_coordinates = self.nodes_array[breakpoint_indicies]
            breakpoint_indicies = breakpoint_indicies[np.argsort(np.linalg.norm(break_point_coordinates - main_branch_min_coordinate, axis=1))]
            for P_j in breakpoint_indicies:  # For all breakpoints P_j in B_i
                P_j_coordinate = self.nodes_array[P_j]
                q_k_node_indices = main_branch_breakpoint_indicies[np.argsort(np.linalg.norm(main_branch_breakpoint_coordinates - P_j_coordinate, axis=1))]
                if len(q_k_node_indices) > 5:
                    q_k_node_indices = q_k_node_indices[:5]
                k, idx, _ = main_branch_kdtree.search_knn_vector_3d(P_j_coordinate, knn=50)
                NP = None
                for i in idx:
                    main_branch_idx = main_branch_node_indices[i]
                    _, abg_candidates = self.compute_bdabg_angles(P_j, main_branch_idx)
                    if np.any(abg_candidates[:, 0] > theta_T):
                        NP = main_branch_idx
                        break

                bdabg_k_candidates = [(q_k_idx, *self.compute_bdabg_angles(P_j, q_k_idx)) for q_k_idx in q_k_node_indices]
                if NP is not None:
                    bda_np_candidates = (NP, *self.compute_bdabg_angles(P_j, NP))

                valid_q_k = [(q_k_idx, bd_k, abg_k) for q_k_idx, bd_k, abg_k in bdabg_k_candidates if np.all([abg_k[:, 0] <= np.cos(np.radians(theta_T)), abg_k[:, 1] <= np.cos(np.radians(theta_T)), abg_k[:, 2] >= np.abs(np.cos(np.radians(theta_T)))])]
                if len(valid_q_k) == 0 and NP is None:
                    continue
                elif len(valid_q_k) == 0:
                    self.set_adjacency(P_j, NP, 1)
                    return True

                q_k_min = min(valid_q_k, key=lambda x: x[1])
                if NP is None:
                    self.set_adjacency(P_j, q_k_min[0], 1)
                    return True
                condition_1 = q_k_min[1] <= 3 * bda_np_candidates[1]
                condition_2 = q_k_min[2][0] <= np.max(bda_np_candidates[2][:, 0])
                if condition_1 and condition_2:
                    self.set_adjacency(P_j, q_k_min[0], 1)
                    return True
                else:
                    self.set_adjacency(P_j, NP, 1)
                    return True

        return False

    def get_parents(self, node_idx):
        """
        Gets the parents of a given node.

        Parameters:
        - node_idx: int
            Index of the node.

        Returns:
        - np.ndarray
            Array of possible parents for the node.
        """
        first_parent_list = self.adjacency_matrix[node_idx].nonzero()[1]
        possible_parents = []
        for first_parent in first_parent_list:
            second_parent_list = self.adjacency_matrix[first_parent].nonzero()[1]
            second_parent_list = second_parent_list[second_parent_list != node_idx]
            for second_parent in second_parent_list:
                possible_parents.append([node_idx, first_parent, second_parent])
        return np.array(possible_parents)

    def compute_bdabg_angles(self, P_j, main_branch_idx):
        """
        Computes the bd, alpha, beta, and gamma angles between a node and its parents.

        Parameters:
        - P_j: int
            Index of the first node.
        - main_branch_idx: int
            Index of the second node.

        Returns:
        - tuple
            Distance between the nodes and array of angles (alpha, beta, gamma).
        """
        bd = np.linalg.norm(self.nodes_array[P_j] - self.nodes_array[main_branch_idx])
        P_j_parent_candidates = self.get_parents(P_j)
        main_branch_parent_candidates = self.get_parents(main_branch_idx)
        abg_candidates = []
        for P_j_parent_candidate in P_j_parent_candidates:
            for main_branch_parent_candidate in main_branch_parent_candidates:
                m_nodes = self.nodes_array[P_j_parent_candidate]
                m_vectors = m_nodes[:2] - m_nodes[1:]
                m = np.mean(m_vectors, axis=0)
                m_normalized = m / np.linalg.norm(m)
                
                n_nodes = self.nodes_array[main_branch_parent_candidate]
                n_vectors = n_nodes[:2] - n_nodes[1:]
                n = np.mean(n_vectors, axis=0)
                n_normalized = n / np.linalg.norm(n)

                l_nodes = np.concatenate((np.flip(m_nodes, axis=0), n_nodes))
                l_vectors = l_nodes[:5] - l_nodes[1:]
                l = np.mean(l_vectors, axis=0)
                l_normalized = l / np.linalg.norm(l)

                cos_alpha = np.dot(m_normalized, n_normalized)
                cos_beta = np.dot(m_normalized, l_normalized)
                cos_gamma = np.dot(n_normalized, l_normalized)
                alpha_deg = np.degrees(np.arccos(cos_alpha))
                beta_deg = np.degrees(np.arccos(cos_beta))
                gamma_deg = np.degrees(np.arccos(cos_gamma))
                
                abg_candidates.append([alpha_deg, beta_deg, gamma_deg])
        return bd, np.array(abg_candidates)

    def get_breakpoints(self, component_id, connected_components):
        """
        Gets the breakpoints for a given component.

        Parameters:
        - component_id: int
            ID of the component.
        - connected_components: np.ndarray
            Array indicating the component assignment of each node.

        Returns:
        - np.ndarray
            Array of breakpoints for the component.
        """
        branch_node_indices = np.argwhere(connected_components[1] == component_id)
        breakpoints_node_indices = [branch_node_idx for branch_node_idx in branch_node_indices if self.adjacency_matrix[branch_node_idx].sum() == 1]
        return np.array(breakpoints_node_indices).flatten()

    def minimum_spanning_tree(self):
        """
        Constructs the minimum spanning tree of the graph.
        """
        fully_connected_weighted_adj_mat = scipy.sparse.lil_matrix((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes - 1):
            for j in range(i + 1, self.num_nodes):
                weight = np.linalg.norm(self.nodes_array[i] - self.nodes_array[j])
                fully_connected_weighted_adj_mat[i, j] = weight
                fully_connected_weighted_adj_mat[j, i] = weight
        G = nx.from_scipy_sparse_array(fully_connected_weighted_adj_mat)
        T = nx.minimum_spanning_tree(G)
        
        mst_adj_mat = scipy.sparse.lil_matrix((self.num_nodes, self.num_nodes))
        for edge in sorted(T.edges(data=True)):
            mst_adj_mat[edge[0], edge[1]] = 1
            mst_adj_mat[edge[1], edge[0]] = 1
        self.adjacency_matrix = mst_adj_mat

class SkeletonMerger:
    def __init__(self, main_tree, likelihood_map, fn_weights):
        """
        Initialize the SkeletonMerger object.

        Parameters:
        - main_tree: UndirectedGraph
            The main tree graph.
        - likelihood_map: UndirectedGraph
            The likelihood map graph.
        - fn_weights: function
            Function to compute weights for the edges.
        """
        self.main_tree = main_tree
        self.likelihood_map = likelihood_map
        self.fn_weights = fn_weights
        for i in range(200):
            print('Path search iter:', i)
            self.associate_nodes_to_main()
            merged = self.merge_shortest_path_components()
            if not merged:
                break

    def associate_nodes_to_main(self):
        """
        Associates nodes in the likelihood map to the main tree components.
        """
        self.nodes_association = np.full(self.likelihood_map.num_nodes, -1)
        main_tree_connected_components = self.main_tree.get_connected_components()
        for map_node_idx, map_node in enumerate(self.likelihood_map.nodes_array):
            k, main_tree_idx, dist_sq = self.main_tree.kdtree.search_radius_vector_3d(map_node, self.main_tree.search_radius) 
            if k > 0:
                self.nodes_association[map_node_idx] = main_tree_connected_components[1][main_tree_idx[0]]
        
        component_min_height_list = []
        for component_id in range(main_tree_connected_components[0]):
            component_indices = np.where(main_tree_connected_components[1] == component_id)[0]
            component_points_array = self.main_tree.nodes_array[component_indices]
            min_height_in_component = np.min(component_points_array[:, 2])
            component_min_height_list.append(min_height_in_component)
        self.component_id_root_order = np.argsort(component_min_height_list)

    def merge_shortest_path_components(self):
        """
        Merges components by finding the shortest path between nodes.

        Returns:
        - bool
            True if components are merged, False otherwise.
        """
        for component_id in self.component_id_root_order:
            root_skeleton_nodes_idx = np.where(self.nodes_association == component_id)[0]
            if len(root_skeleton_nodes_idx) == 0:
                continue
            valid_targets = np.where(np.all([self.nodes_association != component_id, self.nodes_association != -1], axis=0))[0]
            
            nx_graph = nx.from_scipy_sparse_array(self.likelihood_map.adjacency_matrix, create_using=nx.DiGraph())
            path_lengths, paths = nx.multi_source_dijkstra(nx_graph, set(root_skeleton_nodes_idx), target=None, cutoff=None, weight=self.fn_weights)

            shortest_path_length = float('inf')
            for target in valid_targets:
                try:
                    path_length = path_lengths[target]
                except KeyError:
                    path_length = float('inf')
                if path_length < shortest_path_length:
                    shortest_path_length = path_length
                    shortest_target = target
            if shortest_path_length == float('inf'):
                if component_id == self.component_id_root_order[-1]:
                    return False
                else:
                    continue
            shortest_path_idx = paths[shortest_target]
            
            print(f'Shortest Path: Length {shortest_path_length} / Path: {shortest_path_idx}')
            
            shortest_path_nodes = self.likelihood_map.nodes_array[shortest_path_idx]
            head_node, tail_node = shortest_path_nodes[0], shortest_path_nodes[-1]
            k, head_main_tree_idx, _ = self.main_tree.kdtree.search_knn_vector_3d(head_node, 1) 
            k, tail_main_tree_idx, _ = self.main_tree.kdtree.search_knn_vector_3d(tail_node, 1) 
            head_main_tree_idx, tail_main_tree_idx = head_main_tree_idx[0], tail_main_tree_idx[0]
            self.main_tree.bridge_components(shortest_path_nodes, head_main_tree_idx, tail_main_tree_idx)
            return True

    def plot_association(self):
        """
        Plots the association between the skeleton and the main tree.
        """
        skeleton_pcd = o3d.geometry.PointCloud()
        skeleton_pcd.points = o3d.utility.Vector3dVector(self.likelihood_map.nodes_array)    
        skeleton_pcd_colors = np.zeros_like(skeleton_pcd.points)
        no_association_idx = np.where(self.nodes_association == -1)[0]
        association_idx = np.where(self.nodes_association != -1)[0]
        skeleton_pcd_colors[no_association_idx] = np.array([1, 0, 0])
        skeleton_pcd_colors[association_idx] = np.array([0, 0, 1])
        skeleton_pcd.colors = o3d.utility.Vector3dVector(skeleton_pcd_colors)
        main_pcd = self.main_tree.visualize_adjacency_matrix()    
        o3d.visualization.draw_geometries([skeleton_pcd, main_pcd])
        

    def get_main_tree(self):
        """
        Gets the main tree after merging.

        Returns:
        - UndirectedGraph
            The main tree graph.
        """
        return self.main_tree

def point_laplacian_smoothing(pcd, search_radius=0.01):
    """
    Applies Laplacian smoothing to neighbors within a search radius.
    The search radius is computed dynamically based on the aspect ratio of the neighbors.

    Parameters:
    - pcd: open3d.geometry.PointCloud
        The point cloud data.
    - search_radius: float, optional
        The base search radius.

    Returns:
    - open3d.geometry.PointCloud
        The smoothed point cloud.
    """
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    filtered_points = []
    for point in np.asarray(pcd.points):
        k, idx, _ = pcd_tree.search_knn_vector_3d(point, knn=20)
        points_in_radius = np.asarray(pcd.points)[idx, :]

        points_in_radius_centered = points_in_radius - np.mean(points_in_radius, axis=0)
        covariance_matrix = (points_in_radius_centered.T @ points_in_radius_centered) / len(points_in_radius_centered)
        u, s, vh = np.linalg.svd(covariance_matrix)
        AR = np.sqrt(s[1] / s[0])

        k, idx, _ = pcd_tree.search_radius_vector_3d(point, radius=0.001 + search_radius * AR)
        points_in_radius = np.asarray(pcd.points)[idx, :]
        avg_point = np.mean(points_in_radius, axis=0)
        filtered_points.append(avg_point)
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(np.array(filtered_points))
    return filtered_pcd

def equal_spacing(pcd, radiuses):
    """
    Adjusts the spacing between points to be more uniform.

    Parameters:
    - pcd: open3d.geometry.PointCloud
        The point cloud data.
    - radiuses: array-like
        The radius of each node.

    Returns:
    - tuple
        Point cloud with equally spaced points and their corresponding radii.
    """
    unique_points, unique_index = np.unique(np.array(pcd.points), return_index=True, axis=0)
    radiuses = radiuses[unique_index]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(unique_points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    num_points = len(np.asarray(pcd.points))
    total_dist = 0
    for point in np.asarray(pcd.points):
        k, idx, dist = pcd_tree.search_knn_vector_3d(point, 2)
        total_dist += np.sqrt(dist[1])
    radius = total_dist / num_points * 10

    equally_spaced_points = []
    for point in np.asarray(pcd.points):
        k, idx, dist = pcd_tree.search_knn_vector_3d(point, 3)
        neighbor_points = np.asarray(pcd.points)[idx[1:], :] 
        vec1 = neighbor_points[0] - point
        vec2 = neighbor_points[1] - point
        vec1_length = np.linalg.norm(vec1)
        vec2_length = np.linalg.norm(vec2)
        if vec1_length > 0 and vec2_length > 0:
            vec1_normalized = vec1 / vec1_length
            vec2_normalized = vec2 / vec2_length
            cosine_similarity = np.dot(vec1_normalized, vec2_normalized) 
        else: 
            cosine_similarity = 1
        if cosine_similarity < 0 and max(vec1_length, vec2_length) < radius:
            new_point = np.sum(neighbor_points, axis=0) / 2
            equally_spaced_points.append(new_point)
        else:
            equally_spaced_points.append(point)
    
    equally_spaced_pcd = o3d.geometry.PointCloud()
    equally_spaced_pcd.points = o3d.utility.Vector3dVector(np.array(equally_spaced_points))
    equally_spaced_pcd_color = np.zeros_like(equally_spaced_points)
    equally_spaced_pcd_color[:, 1] = 1
    equally_spaced_pcd.colors = o3d.utility.Vector3dVector(equally_spaced_pcd_color)
    return equally_spaced_pcd, radiuses

def construct_initial_skeleton(edges, radius):
    """
    Constructs the initial skeleton from edges and radii.

    Parameters:
    - edges: array-like
        Array of edges.
    - radius: array-like
        Array of radii.

    Returns:
    - UndirectedGraph
        The constructed skeleton.
    """
    edges = np.array(edges)
    edges = np.concatenate(edges)
    edges = edges.reshape(-1, 2, 3)
    spacing = 0.002
    points = []
    radiuses = []
    for edge, r in zip(edges, radius):
        vec = edge[1] - edge[0]
        vec_length = np.linalg.norm(vec) 
        if vec_length == 0:
            points.append(edge[0])
            radiuses.append(r)
        else:
            vec_normalized = vec / vec_length
            for step in np.arange(0, vec_length, spacing):
                point = edge[0] + vec_normalized * step
                points.append(point)
                radiuses.append(r)
    radiuses = np.array(radiuses)
    points_pcd = o3d.geometry.PointCloud()
    points_pcd.points = o3d.utility.Vector3dVector(np.array(points))
    points_color = np.zeros_like(points)
    points_color[:, 2] = 1
    points_pcd.colors = o3d.utility.Vector3dVector(points_color)

    # Apply filtering
    filtered_pcd = copy.deepcopy(points_pcd)
    for i in range(2):
        filtered_pcd = point_laplacian_smoothing(filtered_pcd, search_radius=0.03)

    # Apply equal spacing
    equally_spaced_pcd = copy.deepcopy(filtered_pcd)
    for i in range(10):
        equally_spaced_pcd, radiuses = equal_spacing(equally_spaced_pcd, radiuses)
    
    main_tree = UndirectedGraph(equally_spaced_pcd, radiuses)
    main_tree.construct_initial_graphs()
    main_tree.merge_components()
    return main_tree

def generate_sphere_mesh(pcd, radius):
    """
    Generates a sphere mesh for visualization.

    Parameters:
    - pcd: open3d.geometry.PointCloud
        The point cloud data.
    - radius: array-like
        The radius of each sphere.

    Returns:
    - open3d.geometry.TriangleMesh
        The generated sphere mesh.
    """
    tree_mesh = o3d.geometry.TriangleMesh()
    for point, r in zip(np.array(pcd.points), radius):
        sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=20)
        sphere_mesh.translate(point)
        sphere_mesh.paint_uniform_color([0.588, 0.294, 0])
        tree_mesh += sphere_mesh
    return tree_mesh
