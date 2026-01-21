import numpy as np
import logging

_logger = logging.getLogger(__name__)


def cells_generation(num_cells_x: int, num_cells_y: int, cell_area: float) -> np.ndarray:
    """
    Generates a grid of square cells and computes the center coordinates of each cell.

    Args:
        num_cells_x (int): Number of cells along the x-axis (columns).
        num_cells_y (int): Number of cells along the y-axis (rows).
        cell_area (float): Area of square cell. Assumes square cells (width = height = sqrt(area)).

    Returns:
        np.ndarray: A NumPy array of shape (num_cells_x * num_cells_y, 2) containing (x, y) coordinates of cell centers.
    """
    cell_width = cell_height = np.sqrt(cell_area)
    cells_coords = []
    for row in range(num_cells_y):
        for col in range(num_cells_x):
            x_center = (col + 0.5) * cell_width
            y_center = (row + 0.5) * cell_height
            cells_coords.append([x_center, y_center])
    _logger.info("Created a grid of {} x {} cells (cell area={}km^2)".format(num_cells_x, num_cells_y, cell_area))

    return np.array(cells_coords)


def get_pop_density(population_coords: np.ndarray, high_population_cells: list, min_density: int = 0,
                    max_density: int = 5000) -> np.ndarray:
    """
    Generates random population density values for a list of population grid cell coordinates.

    Args:
        population_coords (np.ndarray): A NumPy array of shape (num_population_cells, 2) containing (x, y) coordinates
            of population cell centers.
        high_population_cells (list): An array of destination cell indices to simulate high population density.
        min_density (int, optional): Minimum population density value.
        max_density (int, optional): Maximum population density value.

    Returns:
        np.ndarray: A NumPy array of population density values (integers) with the same length as `population_coords`.
    """
    _logger.info("Generated random population density for each population grid cell")
    if min_density == max_density:
        return np.array([min_density for _ in range(len(population_coords))])
    else:
        population_density = np.random.randint(min_density, max_density / 10, size=len(population_coords))
        population_density[high_population_cells] = max_density
        return population_density


def get_grid_dimensions(num_cells_x: int, num_cells_y: int, cell_area: float) -> tuple:
    """
    Computes the total width and height of a grid composed of square cells.

    Args:
        num_cells_x (int): Number of cells along the x-axis (columns).
        num_cells_y (int): Number of cells along the y-axis (rows).
        cell_area (float): Area of each square cell. Assumes square cells (width = height = sqrt(area)).

    Returns:
        Tuple:
            float: Total width of the grid.
            float: Total height of the grid.
    """
    cell_width = cell_height = np.sqrt(cell_area)
    total_width = num_cells_x * cell_width
    total_height = num_cells_y * cell_height

    return total_width, total_height


def nodes_generation(num_nodes: int, total_width: int, total_height: int, additional_nodes: list,
                     min_distance_km: int = 30, max_attempts: int = 100) -> np.ndarray:
    """
    Generates a set of 2D nodes within a defined area, ensuring a minimum distance between any two nodes.

    Args:
        num_nodes (int): Number of nodes to generate.
        total_width (float): Width of the area.
        total_height (float): Height of the area.
        additional_nodes (list): A list of nodes coordinates to add to the set.
        min_distance_km (float): Minimum allowed distance between any two nodes (same units as width/height).
        max_attempts (int): Maximum number of attempts per node to find a valid location.

    Returns:
        np.ndarray: A NumPy array of shape (num_nodes, 2) containing (x, y) coordinates of each node.
    """
    coords = []
    attempts = 0
    max_total_attempts = max_attempts * num_nodes

    while len(coords) < num_nodes and attempts < max_total_attempts:
        candidate = np.array([
            np.random.randint(0, total_width),
            np.random.randint(0, total_height)
        ])
        if all(np.linalg.norm(candidate - np.array(c)) >= min_distance_km for c in coords):
            coords.append(candidate)
        attempts += 1

    if len(coords) < num_nodes:
        raise ValueError("Could not place all nodes with given constraints. "
                         "Try reducing 'num_nodes' or 'min_distance_km'.")
    for additional_node in additional_nodes:
        coords.append(additional_node)
    coords = np.array(coords)

    _logger.info("Created {} nodes within an area of {:.1f}km x {:.1f}km with a 'min_distance' between nodes of "
                 "{:.0f}km, added {} nodes based on manual settings".format(num_nodes, total_width,
                                                                            total_height, min_distance_km,
                                                                            len(additional_nodes)))

    return coords


def get_activation_cost_airports(num_airports: int, min_cost: int, max_cost: int) -> np.ndarray:
    """
    Returns a NumPy array of random activation costs for each airport.
    Args:
        num_airports (int): Number of nodes to generate.
        min_cost (int): Minimum activation cost.
        max_cost (int): Maximum activation cost.

    Returns:
        np.ndarray: A NumPy array of activation costs for each airport (based on 'min_cost' and 'max_cost').
    """
    _logger.info(
        "Generated random activation cost for each airport ('min_cost':{}, 'max_cost':{})".format(min_cost, max_cost))
    if min_cost == max_cost:
        return np.array([min_cost for _ in range(num_airports)])
    else:
        return np.random.randint(min_cost, max_cost, size=num_airports)


def get_nodes_distances(nodes_coords: np.ndarray) -> dict:
    """
    Computes the pairwise Euclidean distances between nodes.

    Args:
        nodes_coords (np.ndarray): A NumPy array of shape (n, 2) containing (x, y) coordinates of each node.

    Returns:
        dict: A dictionary where each key is a tuple (i, j) representing a pair of node indices, and the value is the
            Euclidean distance between node i and node j.
    """
    distances = {}
    num_nodes = len(nodes_coords)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = np.linalg.norm(nodes_coords[i] - nodes_coords[j])
            distances[(i, j)] = dist
    _logger.info("Calculated {} pairwise Euclidean distances between {} nodes".format(len(distances), num_nodes))

    return distances


def get_population_cells_near_airports(airports_coords: np.ndarray, population_coords: np.ndarray,
                                       max_ground_distance: float) -> dict:
    """
    Identifies population cells that are within a specified maximum ground distance from each airport.

    Args:
        airports_coords (np.ndarray): A NumPy array of shape (num_airports, 2) containing (x, y) coordinates of each
            airport.
        population_coords (np.ndarray): A NumPy array of shape (num_population_cells, 2) containing (x, y) coordinates
            of population cell centers.
        max_ground_distance (float): Maximum allowed ground distance to consider a population cell "near" an airport.

    Returns:
        dict: A dictionary where each key is an airport index, and each value is a list of indices of population cells
            located within the specified distance from that airport.
    """
    population_cells_near_airports = {}

    for airport_idx, airport_coord in enumerate(airports_coords):
        # Vectorized distance computation
        distances = np.linalg.norm(population_coords - airport_coord, axis=1)
        near_cells = np.where(distances < max_ground_distance)[0].tolist()
        population_cells_near_airports[airport_idx] = near_cells

    _logger.info(
        "Identified all the population cells within a maximum ground distance of {}km from each airport (based "
        "on 'ground_access_config')".format(max_ground_distance))

    return population_cells_near_airports


def get_population_cells2airports_distances(population_coords: np.ndarray, airports_coords: np.ndarray) -> np.ndarray:
    """
    Computes the Euclidean distances between each population cell and each airport.

    Args:
        airports_coords (np.ndarray): A NumPy array of shape (num_airports, 2) containing (x, y) coordinates of each
            airport.
        population_coords (np.ndarray): A NumPy array of shape (num_population_cells, 2) containing (x, y) coordinates
            of population cell centers.

    Returns:
        np.ndarray: A NumPy array of shape (num_population_cells,num_airports) with cells-airports distances.
    """
    diff = population_coords[:, None, :] - airports_coords[None, :, :]

    return np.linalg.norm(diff, axis=2)


def get_destinations_airports_info(destination_cells: list, population_airport_distances: np.ndarray,
                                   max_ground_distance: float) -> list:
    """
    For each destination cell, finds the closest airport if within max_distance.

    Args:
        destination_cells (list): Indices of destination population cells.
        population_airport_distances (np.ndarray): Array of shape (num_population_cells, num_airports) with distances.
        max_ground_distance (float): Maximum allowed ground distance to consider a population cell "near" an airport.

    Returns:
        list: A list containing tuples of the form (destination_cell_idx, closest_airport_idx, distance)
    """
    results = []
    for cell_idx in destination_cells:
        distances = population_airport_distances[cell_idx]

        closest_airport_idx = int(np.argmin(distances))
        closest_distance = float(distances[closest_airport_idx])

        if closest_distance <= max_ground_distance:
            results.append((cell_idx, closest_airport_idx, closest_distance))
            _logger.info("For destination cell {}, the selected destination airports is: {} (closest one within the "
                         "maximum ground distance)".format(cell_idx, closest_airport_idx))
        else:
            results.append((cell_idx, None, None))
            _logger.info(
                "For destination cell {} there are no destination airports within the max ground distance".format(
                    cell_idx))

    return results
