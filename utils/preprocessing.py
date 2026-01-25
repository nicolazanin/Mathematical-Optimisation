import networkx as nx
import numpy as np
import logging

_logger = logging.getLogger(__name__)


def get_threshold_graph(distances: dict, tau: int, mode: str = "below") -> nx.Graph:
    """
    Creates a single undirected graph by including edges based on a distance threshold.

    Args:
        distances (dict): A dictionary where each key is a tuple (i, j) representing a pair of node indices, and the
            value is the Euclidean distance between node i and node j.
        tau (int): Distance threshold for including edges.
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
    _logger.info("Created graph with distance threshold {} tau, with {} edges and {} nodes (tau={}km)".
                 format(mode, len(graph.edges), len(graph.nodes), tau))

    return graph


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
            paths = nx.all_simple_paths(graph, source=source_node, target=destination_airports, cutoff=max_path_edges)
            all_paths.extend(list(paths))
    _logger.info("Defined {} simple paths to destination airport/s: {}".format(len(all_paths),
                                                                               destination_airports))

    return np.array(all_paths, dtype=object)


def get_attractive_paths_from_rft(paths: np.ndarray, distances: dict, routing_factor_thr: float) -> np.ndarray:
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
            node_pair = tuple((path[i], path[i + 1]))
            total_distance += distances[node_pair]
        direct_pair = tuple((path[0], path[-1]))
        direct_distance = distances[direct_pair]

        routing_factor = total_distance / direct_distance

        if routing_factor <= routing_factor_thr:
            attractive_paths.append(path)

    _logger.info("Defined {} attractive paths based on the routing factor threshold (routing_factor_thr={})".format(
        len(attractive_paths), routing_factor_thr))

    return np.array(attractive_paths, dtype=object)


def get_population_cells_paths(population_coords, paths: np.ndarray, distances: dict,
                               population_cells_near_airports: dict, destinations_airports_info: list,
                               population_cells2airport_distances: np.ndarray,
                               population_cells_too_close_to_destination_cells: dict,
                               airports_too_close_to_destination_cells: dict, ground_speed: int, air_speed: int,
                               max_total_time: float) -> dict:
    """
    Determine feasible flight paths for each population cell based on travel-time constraints.

    For each population cell, this function identifies all flight paths that:
    - start at an airport located near the population cell,
    - end at an airport associated with a destination population cell,
    - do not connect population cells that are too close to a destination cell,
    - do not connect airports that are too close to a destination cell,
    - and satisfy a maximum total travel time constraint that includes:
        * ground travel from the population cell to the departure airport,
        * air travel along the selected flight path,
        * ground travel from the arrival airport to the destination cell.

    Args:
        population_coords (np.ndarray): A NumPy array of shape (num_population_cells, 2) containing (x, y) coordinates
            of population cell centers.
        paths (np.ndarray): A NumPy array of paths (each path is a list of node IDs).
        distances (dict): A dictionary where each key is a tuple (i, j) representing a pair of nodes indices, and the
            value is the Euclidean distance between node i and node j.
        population_cells_near_airports (dict): A dictionary where each key is an airport index, and each value is a list
            of indices of population cells located within the specified distance from that airport.
        destinations_airports_info (list): Each tuple in the list contains: (destination_cell_idx, closest_airport_idx,
            distance)
        population_cells2airport_distances (np.ndarray): Array of shape (num_population_cells,num_airports) with
            cells-airports distances.
        population_cells_too_close_to_destination_cells (dict): A dictionary mapping each destination cell to an array of
            population cell indices that are too close to it.
        airports_too_close_to_destination_cells (dict): A dictionary mapping each destination cell to an array of
            airports indices that are too close to it.
        ground_speed (int): Average ground speed.
        air_speed (int): Aircraft cruise speed.
        max_total_time (float): Maximum total travel time threshold (ground access component + flight time)

    Returns:
        dict: A dictionary mapping each population cell index to a list of paths (each path is a list of node IDs)
            starting from an airport near that population cell.
    """
    dest_airport_info = {
        airport_idx: {
            "dest_cell": dest_cell,
            "distance": distance,
        }
        for dest_cell, airport_idx, distance in destinations_airports_info}
    population_cells_paths = {idx: [] for idx in range(len(population_coords))}

    for simple_path in paths:
        start_airport = simple_path[0]
        end_airport = simple_path[-1]
        dest_cell = dest_airport_info[end_airport]["dest_cell"]
        if start_airport not in airports_too_close_to_destination_cells[dest_cell]:
            flight_distance = 0.0
            for i in range(len(simple_path) - 1):
                node_pair = tuple((simple_path[i], simple_path[i + 1]))
                flight_distance += distances[node_pair]

            flight_time = flight_distance / air_speed

            for pop in population_cells_near_airports[start_airport]:
                if pop not in np.append(population_cells_too_close_to_destination_cells[dest_cell], dest_cell):
                    ground_start = population_cells2airport_distances[pop, start_airport] / ground_speed
                    ground_end = dest_airport_info[end_airport]["distance"] / ground_speed

                    total_time = ground_start + flight_time + ground_end

                    if total_time <= max_total_time:
                        population_cells_paths[pop].append(simple_path)
                    else:
                        _logger.debug("The path {} can not be used by population cell {} due to "
                                      "the limit on the 'max_total_time_travel' of {} hours".format(
                            simple_path, pop, max_total_time))
                else:
                    _logger.debug(
                        "The path {} can not be used by population cell {} because the area is too close to the "
                        "destination cell {} (based on 'min_ground_travel_time_to_destination_cell' and"
                        " ground 'avg_speed')".format(simple_path, pop, dest_cell))
        else:
            _logger.debug("The path {} can not be used because the starting airport: {} is too close to the "
                          "destination cell {} (based on 'min_ground_travel_time_to_destination_cell' and"
                          " ground 'avg_speed')".format(simple_path, simple_path[0], dest_cell))

    return population_cells_paths


def get_population_cells_too_close_to_destination_cells(population_coords: np.ndarray, destination_cells: np.ndarray,
                                                        min_distance_to_destination_cells: float) -> dict:
    """
    For each target population cell, identifies the population cells that are closer than a minimum distance.

    Args:
        population_coords (np.ndarray): A NumPy array of shape (num_population_cells, 2) containing (x, y) coordinates
            of population cell centers.
        destination_cells (list): Indices of destination population cells.
        min_distance_to_destination_cells(float): Minimum allowed distance to consider a population cell or an airport
            from the destination cells (km).
    Returns:
        dict: A dictionary mapping each destination cell to an array of population cell indices that are too close to it.
    """
    target_coords = population_coords[destination_cells]
    diff = target_coords[:, None, :] - population_coords[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    distances[np.arange(len(destination_cells)), destination_cells] = np.inf  # self-distances ingnore

    target_to_close_cells = {}

    for i, target_idx in enumerate(destination_cells):
        close_cells = np.where(distances[i] < min_distance_to_destination_cells)[0]
        target_to_close_cells[target_idx] = close_cells
        _logger.info(
            "For the destination cell {} the following population cells {} are closer than the minimum distance of {:.0f}km".format(
                target_idx, close_cells, min_distance_to_destination_cells))

    return target_to_close_cells


def get_airports_too_close_to_destination_cells(airports_coords: np.ndarray, population_coords: np.ndarray,
                                                destination_cells: np.ndarray,
                                                min_distance_to_destination_cells: float) -> dict:
    """
    For each target population cell, identifies the airports that are closer than a minimum distance.

    Args:
        airports_coords (np.ndarray): A NumPy array of shape (num_airports, 2) containing (x, y) coordinates of each
            airport.
        population_coords (np.ndarray): A NumPy array of shape (num_population_cells, 2) containing (x, y) coordinates
            of population cell centers.
        destination_cells (list): Indices of destination population cells.
        min_distance_to_destination_cells(float): Minimum allowed distance to consider a population cell or an airport
            from the destination cells (km).

    Returns:
        dict: A dictionary mapping each destination cell to an array of airports indices that are too close to it.
    """
    target_coords = population_coords[destination_cells]
    diff = target_coords[:, None, :] - airports_coords[None, :, :]
    distances = np.linalg.norm(diff, axis=2)

    target_to_close_airports = {}

    for i, target_idx in enumerate(destination_cells):
        close_cells = np.where(distances[i] < min_distance_to_destination_cells)[0]
        target_to_close_airports[target_idx] = close_cells
        _logger.info(
            "For the destination cell {} the following airports {} are closer than the minimum distance of {:.0f}km".format(
                target_idx, close_cells, min_distance_to_destination_cells))

    return target_to_close_airports


def get_attractive_paths(population_cells_paths: dict) -> np.ndarray:
    """
    For each target population cell, identifies the population cells that are closer than a minimum distance.

    Args:
        population_cells_paths (dict): A dictionary mapping each population cell index to a list of paths (each path is
            a list of node IDs) starting from an airport near that population cell.

    Returns:
         np.ndarray: A NumPy array of attractive paths (each path is a list of node IDs).
    """
    attractive_paths = list({tuple(lst) for lists in population_cells_paths.values() for lst in lists})
    _logger.info("Defined {} attractive paths".format(len(attractive_paths)))

    return np.array([list(t) for t in attractive_paths], dtype=object)


def get_attractive_graph(distances: dict, attractive_paths: np.ndarray) -> nx.Graph:
    """
    Creates a single undirected graph by including edges based on the attractive paths.

    Args:
        distances (dict): A dictionary where each key is a tuple (i, j) representing a pair of nodes indices, and the
            value is the Euclidean distance between node i and node j.
        attractive_paths (np.ndarray): A NumPy array of attractive paths (each path is a list of node IDs).

    Returns:
        nx.Graph: A NetworkX graph containing the filtered edges from the attractive paths.
    """
    active_graph = nx.Graph()
    for path in attractive_paths:
        for i in range(len(path) - 1):
            node_pair = tuple((path[i], path[i + 1]))
            active_graph.add_edge(path[i], path[i + 1], weight=distances[node_pair])
    _logger.info(
        "Created attractive graph with {} edges and {} nodes".format(len(active_graph.edges), len(active_graph.nodes)))

    return active_graph
