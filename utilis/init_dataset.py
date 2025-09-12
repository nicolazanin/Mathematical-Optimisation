import numpy as np

np.random.seed(67) # :)

def nodes_generation(num_nodes: int,
                     total_width: int,
                     total_height: int,
                     min_distance_km: int = 30,
                     max_attempts: int = 100) -> np.ndarray:
    """
    Generates a set of 2D nodes within a defined area, ensuring a minimum distance between any two nodes.

    Args:
        num_nodes (int): Number of nodes to generate.
        total_width (float): Width of the area.
        total_height (float): Height of the area.
        min_distance_km (float): Minimum allowed distance between any two nodes (same units as width/height).
        max_attempts (int): Maximum number of attempts per node to find a valid location.

    Returns:
        np.ndarray: A NumPy array of shape (num_nodes, 2) containing the (x, y) coordinates of each node.
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

    coords = np.array(coords)

    return coords

def nodes_distances(nodes_coords: np.ndarray) -> dict:
    """
    Computes the pairwise Euclidean distances between nodes.

    Args:
        nodes_coords (np.ndarray): A NumPy array of shape (n, 2) containing the (x, y) coordinates of each node.

    Returns:
        dict: A dictionary where each key is a tuple (i, j) representing a pair of node indices,
        and the value is the Euclidean distance between node i and node j.
    """
    distances = {}
    num_nodes = len(nodes_coords)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = np.linalg.norm(nodes_coords[i] - nodes_coords[j])
            distances[(i, j)] = dist

    return distances


def cells_generation(num_cells_x: int, num_cells_y: int, cell_area: float) -> np.ndarray:
    """
    Generates a grid of square cells and computes the center coordinates of each cell.

    Args:
        num_cells_x (int): Number of cells along the x-axis (columns).
        num_cells_y (int): Number of cells along the y-axis (rows).
        cell_area (float): Area of each square cell. Assumes square cells (width = height = sqrt(area)).

    Returns:
        np.ndarray: A NumPy array of shape (num_cells_x * num_cells_y, 2) containing (x, y) center coordinates.
    """
    cell_width = cell_height = np.sqrt(cell_area)
    cells_coords = []
    for row in range(num_cells_y):
        for col in range(num_cells_x):
            x_center = (col + 0.5) * cell_width
            y_center = (row + 0.5) * cell_height
            cells_coords.append([x_center, y_center])

    return np.array(cells_coords)

def grid_dimensions(num_cells_x: int, num_cells_y: int, cell_area: float) -> tuple:
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


def get_pop_cells_near_airports(airports_coords: np.ndarray,
                                 pop_coords: np.ndarray,
                                 max_ground_distance: float) -> dict:
    """
    Identifies population cells that are within a specified maximum ground distance from each airport.

    Args:
        airports_coords (np.ndarray): Array of shape (num_airports, 2) containing (x, y) coordinates of airports.
        pop_coords (np.ndarray): Array of shape (num_population_cells, 2) containing (x, y) coordinates of population cell centers.
        max_ground_distance (float): Maximum allowed ground distance to consider a population cell "near" an airport.

    Returns:
        dict: A dictionary where each key is an airport index, and each value is a list of indices
        of population cells located within the specified distance from that airport.
    """
    pop_cells_near_airports = {}

    for airport_idx, airport_coord in enumerate(airports_coords):
        # Vectorized distance computation
        distances = np.linalg.norm(pop_coords - airport_coord, axis=1)
        near_cells = np.where(distances < max_ground_distance)[0].tolist()
        pop_cells_near_airports[airport_idx] = near_cells

    return pop_cells_near_airports

def get_pop_density(pop_coords: np.ndarray, min_density: int = 0, max_density: int = 50000) -> np.ndarray:
    """
    Generates random population density values for a list of population cell coordinates.

    Args:
        pop_coords (np.ndarray): Array of shape (num_population_cells, 2) containing (x, y) coordinates of population cell centers.
        min_density (int, optional): Minimum population density value. Defaults to 0.
        max_density (int, optional): Maximum population density value. Defaults to 50,000.

    Returns:
        np.ndarray: Array of population density values (integers) with the same length as `pop_coords`.
    """
    return np.random.randint(min_density, max_density, size=len(pop_coords))
