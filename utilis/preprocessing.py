import networkx as nx
import numpy as np
import logging

_logger = logging.getLogger(__name__)


def  create_threshold_graph(distances: dict, tau: float, mode: str = "below") -> nx.Graph:
    """
    Creates a single undirected graph by including edges based on a distance threshold.

    Args:
        distances (dict): A dictionary where each key is a tuple (i, j) representing a pair of node indices, and the
        value is the Euclidean distance between node i and node j.
        tau (float): Distance threshold for including edges.
        mode (str): Determines which edges to include:
            - "below": includes edges with distance <= tau.
            - "above": includes edges with distance > tau.

    Returns:
        nx.Graph: A NetworkX graph containing the filtered edges.
    """
    graph = nx.Graph()

    for (i, j), d in distances.items():
        if (mode == "below" and d <= tau) or (mode == "above" and d > tau):
            graph.add_edge(i, j, weight=d)
    _logger.info("Created graph with distance threshold {} tau, {} edges and {} nodes (tau={}km)".
                 format(mode, len(graph.edges), len(graph.nodes), tau))

    return graph


def get_attractive_paths(paths: np.ndarray, distances: dict, routing_factor_thr: float) -> np.ndarray:
    """
    Removes paths that are considered unattractive based on the routing factor threshold.

    A path is considered unattractive if the ratio between the path’s total distance and the nonstop distance between
    the origin and destination node exceeds the given `routing_factor_thr`. Only paths with routing factor less than or
    equal to the threshold are returned.

    Args:
        paths (list): A NumPy array of all simple paths (each path is a list of node IDs).
        distances (dict): A dictionary where each key is a tuple (i, j) representing a pair of nodes indices, and the
        value is the Euclidean distance between node i and node j.
        routing_factor_thr (float): Ratio between the path’s total distance and the nonstop distance

    Returns:
         np.ndarray: A NumPy array of attractive paths (each path is a list of node IDs).
    """
    attractive_paths = []
    for path in paths:
        total_distance = 0.0
        for i in range(len(path) - 1):
            if path[i] < path[i + 1]:
                node_pair = (path[i], path[i + 1])
            else:
                node_pair = (path[i + 1], path[i])

            total_distance += distances[node_pair]
        if path[0] < path[-1]:
            direct_pair = (path[0], path[-1])
        else:
            direct_pair = (path[-1], path[0])

        direct_distance = distances[direct_pair]

        routing_factor = total_distance / direct_distance

        if routing_factor <= routing_factor_thr:
            attractive_paths.append(path)

    _logger.info("Removed {} unattractive based on the routing factor threshold (routing_factor_thr={})".format(
        len(paths) - len(attractive_paths), routing_factor_thr))

    return np.array(attractive_paths, dtype=object)


def get_all_paths_to_destinations(graph: nx.Graph, destination_airports: np.ndarray,
                                         max_path_edges: int) -> np.ndarray:
    """
    Finds all paths from all non-destination airport nodes (path origin) to any of the destination nodes in a graph, up
    to a specified maximum path length.

    Args:
        graph (nx.Graph): A NetworkX graph.
        destination_airports (np.ndarray): A NumPy array of destination airports indices close to the destination cells.
        max_path_edges (int): Maximum allowed path edges.

    Returns:
        np.ndarray: A NumPy array of all paths (each path is a list of node IDs).
    """
    # Paths are only computed from source nodes that are not in the destination set.
    all_paths = []
    for source_node in graph.nodes():
        if source_node not in destination_airports:
            paths = nx.all_simple_paths(graph, source=source_node, target=destination_airports,
                                        cutoff=max_path_edges)
            all_paths.extend(list(paths))

    return np.array(all_paths, dtype=object)


def get_population_cells_paths(population_coords, paths: np.ndarray,
                               population_cells_near_airports: dict) -> dict:
    """
    Finds for each population cell the set of paths (each path is a list of node IDs) starting from an airport near that
    population cell.

    Args:
        population_coords (np.ndarray): A NumPy array of shape (num_population_cells, 2) containing (x, y) coordinates
        of population cell centers.
        paths (np.ndarray): A NumPy array of paths (each path is a list of node IDs).
        population_cells_near_airports (dict): A dictionary where each key is an airport index, and each value is a list
        of indices of population cells located within the specified distance from that airport.

    Returns:
        dict: A dictionary mapping each population cell index to a list of paths (each path is a list of node
        IDs) starting from an airport near that population cell.
    """
    population_cells_paths = {}
    for idx, pop in enumerate(population_coords):
        population_cells_paths[idx] = []

    for simple_path in paths:
        starting_airport = simple_path[0]
        for pop in population_cells_near_airports[starting_airport]:
            population_cells_paths[pop].append(simple_path)

    return population_cells_paths
