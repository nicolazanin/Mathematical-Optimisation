import random
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
from collections import defaultdict


def model(airports: list, paths: np.ndarray, graph: nx.Graph, population_cells_paths: dict,
          destinations_airports_info: list, tau: int, mip_gap: float, epsilon: int, max_run_time: int) -> tuple:
    """

    Args:
        airports (list): List of airport IDs.
        paths (np.ndarray): A NumPy array of paths (each path is a list of node IDs).
        graph (nx.Graph): A NetworkX graph containing the paths.
        population_cells_paths (dict): A dictionary mapping each population cell index to a list of paths (each path is
            a list of node IDs) starting from an airport near that population cell.
        destinations_airports_info (list): Each tuple in the list contains: (destination_cell_idx, closest_airport_idx,
            distance)
        tau (int): Maximum travel range on a single charge.
        mip_gap (float): MIP gap termination condition value.
        epsilon (int): Small positive number to define big-M parameters.
        max_run_time (int): The maximum run time in seconds.

    Returns:
        tuple: A tuple containing the model, variables y and variables phi.

    """
    m = gp.Model("eanc")
    m.setParam('LogToConsole', 0)
    m.setParam('MIPGap', mip_gap)
    m.setParam('TimeLimit', max_run_time)
    m1_vals, m2_vals, m3_vals = get_tight_big_m(graph=graph, tau=tau, epsilon=epsilon)
    y = m.addVars([i for i in airports], vtype=GRB.BINARY, name="y")
    rho = m.addVars([i for i in airports], vtype=GRB.CONTINUOUS, name="rho", lb=0.0)
    chi = m.addVars([i for i in airports], vtype=GRB.BINARY, name="chi")

    canonical_edges = sorted(list(tuple(sorted(edge)) for edge in graph.edges()))
    z = m.addVars(canonical_edges, vtype=GRB.BINARY, name="z")

    w = m.addVars([(i, j) for i in airports for j in graph.neighbors(i)],
                  vtype=GRB.BINARY, name="w")

    dest_airports_info = {dest_cell: airport_idx for dest_cell, airport_idx, _ in destinations_airports_info}

    psi = m.addVars(len(paths), vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="psi")
    phi = m.addVars([(i, j) for i in population_cells_paths.keys() for j in dest_airports_info.keys()],
                    vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="phi")
    for airport in airports:
        neighbors = list(graph.neighbors(airport))
        m.addConstr(rho[airport] <= m1_vals[airport] * (1 - y[airport]))  # 12
        m.addConstr(y[airport] + gp.quicksum(w[(airport, neighbor)] for neighbor in neighbors) +
                    chi[airport] == 1)  # 16
        m.addConstr(rho[airport] >= m1_vals[airport] * chi[airport])  # 15
        for neighbor in neighbors:
            m.addConstr(rho[airport] <= graph.edges[airport, neighbor]['weight'] + rho[neighbor])  # 13
            m.addConstr(rho[airport] >= graph.edges[airport, neighbor]['weight'] + rho[neighbor] - m2_vals[
                (airport, neighbor)] * (
                                1 - w[(airport, neighbor)]))  # 14

        min_dist_to_neighbor = min(graph.edges[airport, neighbor]['weight'] for neighbor in neighbors)
        m.addConstr(rho[airport] >= min(min_dist_to_neighbor, m1_vals[airport]) * (1 - y[airport]))  # 20

    for airport_i, airport_j in canonical_edges:
        m.addConstr(graph.edges[airport_i, airport_j]['weight'] + rho[airport_i] + rho[airport_j] <= tau +
                    m3_vals[airport_i, airport_j] * (1 - z[airport_i, airport_j]))  # 4
        m.addConstr(z[airport_i, airport_j] <= 1 - chi[airport_i])  # 21
        m.addConstr(z[airport_i, airport_j] <= 1 - chi[airport_j])  # 22
        m4_ij = tau - graph.edges[airport_i, airport_j]['weight']
        m.addConstr(graph.edges[airport_i, airport_j]['weight'] + rho[airport_i] + rho[airport_j] + m4_ij *
                    z[airport_i, airport_j] >= tau)  # 24

    for idx, path in enumerate(paths):
        path_edges = [tuple(sorted((path[i], path[i + 1]))) for i in range(len(path) - 1)]
        m.addConstr(psi[idx] - gp.quicksum(z[edge] for edge in path_edges) >= 1 - len(path_edges))  # 23
        for edge in path_edges:
            m.addConstr(psi[idx] <= z[edge])  # 5

    for pop_cell in population_cells_paths:
        for dest_cell, airport_idx, _ in destinations_airports_info:
            pop_cell_paths_to_airport = [path for path in population_cells_paths[pop_cell] if path[-1] == airport_idx]
            path_indices = [idx for idx, path in enumerate(paths) if path in pop_cell_paths_to_airport]
            if path_indices:
                m.addConstr(phi[pop_cell, dest_cell] <= gp.quicksum(psi[p_id] for p_id in path_indices))
            else:
                m.addConstr(phi[pop_cell, dest_cell] == 0)
    m.update()

    return m, y, phi


def get_tight_big_m(graph, tau, epsilon) -> tuple:
    """
    Calculates the tight big-M in order to tighten the model relaxation and accelerate the convergence of branch-and-cut
    algorithms.

    Args:
        graph (nx.Graph): A NetworkX graph containing the paths.
        tau (int): Maximum travel range on a single charge.
        epsilon (int): Small positive number to define big-M parameters.

    Returns:
        tuple: A tuple of 3 dictionaries (for big-M 1, 2, 3) where each key is a tuple (i, j) representing a pair of
            nodes indices, and the value is the big-M parameter between node i and node j.
    """
    m1_vals = {}
    for airport in list(graph.nodes):
        neighbors = list(graph.neighbors(airport))
        min_dist_to_neighbor = min(graph.edges[airport, neighbor]['weight'] for neighbor in neighbors)
        m1_vals[airport] = tau - min_dist_to_neighbor + epsilon

    m2_vals = {}
    m3_vals = {}
    for i, j in graph.edges():
        neighbors_i = list(graph.neighbors(i))
        neighbors_j = list(graph.neighbors(j))

        min_dist_from_j = min(graph.edges[j, neighbor]['weight'] for neighbor in neighbors_j)
        min_dist_from_i = min(graph.edges[i, neighbor]['weight'] for neighbor in neighbors_i)

        m2_vals[(i, j)] = (graph.edges[i, j]['weight'] + tau - min_dist_from_j + epsilon)
        m3_vals[(i, j)] = (graph.edges[i, j]['weight'] + tau - min_dist_from_i - min_dist_from_j + epsilon * 2)
        m2_vals[(j, i)] = (graph.edges[j, i]['weight'] + tau - min_dist_from_i + epsilon)
        m3_vals[(j, i)] = (graph.edges[j, i]['weight'] + tau - min_dist_from_j - min_dist_from_i + epsilon * 2)

    return m1_vals, m2_vals, m3_vals


def get_initial_kernel(population_cells_paths: dict, initial_kernel_size: int) -> list:
    """
    Generates the initial kernel for the KS heuristic. For each airport nodes computes the number of population
    cells that could be served based on the paths that go through it, then ranks the airports in descending order and
    selects first part based on the initial kernel size.

    Args:
        population_cells_paths (dict): A dictionary mapping each population cell index to a list of paths (each path is
            a list of node IDs) starting from an airport near that population cell.
        initial_kernel_size (int): # Kernel search heuristic initial kernel size.

    Returns:
        list: A list of airports nodes IDs chosen for the initial kernel.
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


def get_buckets(airports: list, kernel: list, bucket_size: int) -> dict:
    """
    Divides the set of non-kernel airports into buckets.

    Args:
        airports (list): List of airport IDs.
        kernel (list): Current kernel
        bucket_size (int): The size of the buckets we want to use.

    Returns:
        dict: A dictionary where each key is a bucket index, and each value is a list of airport IDs representing the
            bucket.
    """
    not_kernel = list(set(airports) - set(kernel))
    random.shuffle(not_kernel)
    buckets = {i: not_kernel[i * bucket_size:(i + 1) * bucket_size] for i in range((len(not_kernel) + bucket_size - 1)
                                                                                   // bucket_size)}

    return buckets


def get_outputs_from_model(m):
    y, psi, phi, rho, chi, z, w = get_model_variables(m)

    charging_airports = [int(name[2:-1]) for name, value in y.items() if value == 1]
    population_covered = sorted([eval(name[4:-1])[0] for name, value in phi.items() if value > 0.99])
    active_path_indices = np.array([int(name[4:-1]) for name, value in psi.items() if value == 1])

    return charging_airports, population_covered, active_path_indices, m.ObjBound


def get_model_variables(m: gp.Model) -> tuple:
    """
    Extract variables from the model.

    Args:
        m (gp.Model): The Gurobi model to extract variables from.

    Returns:
        tuple: A tuple containing lists of model variables.
    """
    all_vars = m.getVars()
    y_vars = [v for v in all_vars if v.VarName.startswith('y')]
    psi_vars = [v for v in all_vars if v.VarName.startswith('psi')]
    phi_vars = [v for v in all_vars if v.VarName.startswith('phi')]
    rho_vars = [v for v in all_vars if v.VarName.startswith('rho')]
    chi_vars = [v for v in all_vars if v.VarName.startswith('chi')]
    z_vars = [v for v in all_vars if v.VarName.startswith('z')]
    w_vars = [v for v in all_vars if v.VarName.startswith('w')]

    psi = vars_to_dict(psi_vars)
    y = vars_to_dict(y_vars)
    phi = vars_to_dict(phi_vars)
    rho = vars_to_dict(rho_vars)
    chi = vars_to_dict(chi_vars)
    z = vars_to_dict(z_vars)
    w = vars_to_dict(w_vars)

    return y, psi, phi, rho, chi, z, w


def vars_to_dict(var_list):
    return {v.VarName: v.X for v in var_list }


