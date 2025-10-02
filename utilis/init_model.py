import networkx as nx
import logging

_logger = logging.getLogger(__name__)


def get_all_simple_path_to_destinations(graph: nx.Graph, destination_nodes: list, max_path_edges: int) -> list:
    """
    Finds all simple paths from all non-destination nodes to any of the destination nodes in a graph, up to a specified
    maximum path length.

    Args:
        graph (nx.Graph): A NetworkX graph.
        destination_nodes (list): A list of nodes considered as destination nodes.
        max_path_edges (int): Maximum allowed path edges.

    Returns:
        list: An array of all simple paths (each path is a list of node IDs).
    """
    # Paths are only computed from source nodes that are not in the destination set.
    all_simple_paths = []
    for source_node in graph.nodes():
        if source_node not in destination_nodes:
            # Find all simple paths from the source node to any destination node
            paths = nx.all_simple_paths(graph, source=source_node, target=destination_nodes,
                                        cutoff=max_path_edges)
            all_simple_paths.extend(list(paths))

    return all_simple_paths

def get_pop_paths(pop_coords, all_simple_paths: list, pop_cells_near_airports) -> dict:
    """
    Args:
        pop_coords (np.ndarray): Array of shape (num_population_cells, 2) containing (x, y) coordinates of population cell centers.
        all_simple_paths (list): An array of all simple paths (each path is a list of node IDs).
        pop_cells_near_airports (dict): A dictionary where each key is an airport index, and each value is a list of indices
        of population cells located within the specified distance from that airport.
    Returns:
        dict: A dictionary mapping each population cell index to a list of simple paths (each path is a list of node IDs)
        starting from an airport near that population cell.
    """
    pop_paths = {}
    for idx, pop in enumerate(pop_coords):
        pop_paths[idx] = []

    for simple_path in all_simple_paths:
        starting_airport = simple_path[0]
        for pop in pop_cells_near_airports[starting_airport]:
            pop_paths[pop].append(simple_path)

    return pop_paths
