from collections import defaultdict
import random


def calculate_tight_big_m(active_graph, tau, epsilon) -> tuple:
    """
    Returns the tight big m in order to tighten the model relaxation and accelerate the convergence of branch-and-cut
    algorithms.

    Args:
        active_graph (nx.Graph): A NetworkX active airports graph.
        tau (int): Maximum travel range on a single charge
        epsilon (int): Small positive number
    Returns:
        tuple: Big M parameters.
    """
    m1_vals = {}
    for airport in list(active_graph.nodes):
        neighbors = list(active_graph.neighbors(airport))
        min_dist_to_neighbor = min(active_graph.edges[airport, neighbor]['weight'] for neighbor in neighbors)
        m1_vals[airport] = tau - min_dist_to_neighbor + epsilon

    m2_vals = {}
    m3_vals = {}
    for i, j in active_graph.edges():
        edge = tuple(sorted((i, j)))

        neighbors_i = list(active_graph.neighbors(edge[0]))
        neighbors_j = list(active_graph.neighbors(edge[1]))

        min_dist_from_j = min(active_graph.edges[edge[1], neighbor]['weight'] for neighbor in neighbors_j)
        min_dist_from_i = min(active_graph.edges[edge[0], neighbor]['weight'] for neighbor in neighbors_i)

        m2_vals[edge] = (
                active_graph.edges[edge[0], edge[1]]['weight'] + tau - min_dist_from_j + epsilon)
        m3_vals[edge] = (
                active_graph.edges[edge[0], edge[1]]['weight'] + tau - min_dist_from_i - min_dist_from_j + epsilon * 2)

    return m1_vals, m2_vals, m3_vals


def get_initial_kernel(population_cells_paths, initial_kernel_size) -> list:
    """
    Return the initial kernel for the EACN-KS heuristic. For each airport nodes computes the number of population cells
    that could be served based on the paths that go through it, then ranks the airports in descending order and selects
    first part based on the initial kernel size.
    Args:
        population_cells_paths (dict): Dictionary mapping each population cell index to a list of paths (each path is a list of node
        IDs) starting from an airport near that population cell.
        initial_kernel_size (int): The initial kernel size.
    Returns:
        initial_kernel (list): List of airports nodes IDs chosen for the initial kernel.
    """

    airport_served_pops = defaultdict(list)
    for pop_id, paths in population_cells_paths.items():
        for path in paths:
            for airport_node in path:
                airport_served_pops[airport_node].append(pop_id)

    airport_scores = {}
    for airport_id, population_cells_served in airport_served_pops.items():
        score = len(population_cells_served)
        airport_scores[airport_id] = score

    sorted_airports = sorted(airport_scores.items(), key=lambda item: item[1], reverse=True)

    initial_kernel_with_scores = sorted_airports[:initial_kernel_size]
    initial_kernel = [airport_id for airport_id, score in initial_kernel_with_scores]

    return initial_kernel


def get_buckets(airports, kernel, bucket_size) -> dict:
    """

    """
    not_kernel = [airport for airport in airports if airport not in kernel]
    random.shuffle(not_kernel)
    num_backets = len(not_kernel) // bucket_size
    buckets = {}
    for i in range(num_backets):
        if (i + 1) * bucket_size > len(not_kernel):
            end = len(not_kernel)
        else:
            end = (i + 1) * bucket_size
        buckets[i] = not_kernel[i * bucket_size:end]
    return buckets
