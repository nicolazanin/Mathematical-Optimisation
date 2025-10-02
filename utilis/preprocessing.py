import networkx as nx
import logging

_logger = logging.getLogger(__name__)


def create_threshold_graph(distances: dict, tau: float, mode: str = "below") -> nx.Graph:
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


def get_attractive_paths(paths: list, distances: dict, routing_factor_thr: float) -> list:
    """
    Removes paths that are considered unattractive based on the routing factor threshold.

    A path is considered unattractive if the ratio between the path’s total distance and the nonstop distance between
    the origin and destination node exceeds the given `routing_factor_thr`. Only paths with routing factor less than or
    equal to the threshold are returned.

    Args:
        paths (list): An array of all simple paths (each path is a list of node IDs).
        distances (dict): A dictionary where each key is a tuple (i, j) representing a pair of nodes indices, and the
        value is the Euclidean distance between node i and node j.
        routing_factor_thr (float): Ratio between the path’s total distance and the nonstop distance

    Returns:
        list: An array of all simple paths without the unattractive paths (each path is a list of node IDs).
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

    return attractive_paths
