import random
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
from utils.settings import settings


def model(airports, paths, graph, population_cells_paths, destination_cells2destination_airports):
    m = gp.Model("aa")
    m.setParam('LogToConsole', 0)
    m1_vals, m2_vals, m3_vals = calculate_tight_big_m(active_graph=graph,
                                                      tau=settings.aircraft_config.tau,
                                                      epsilon=settings.model_config.epsilon)
    y = m.addVars([i for i in airports], vtype=GRB.BINARY, name="y")
    rho = m.addVars([i for i in airports], vtype=GRB.CONTINUOUS, name="rho", lb=0.0)
    w = m.addVars([(i, j) for i in airports for j in graph.neighbors(i)],
                  vtype=GRB.BINARY, name="w")
    chi = m.addVars([i for i in airports], vtype=GRB.BINARY, name="chi")

    canonical_edges = sorted(list(set(tuple(sorted(edge)) for edge in graph.edges())))

    z = m.addVars(canonical_edges, vtype=GRB.BINARY, name="z")

    psi = m.addVars(len(paths), vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="psi")
    phi = m.addVars([(i, j) for i in range(len(population_cells_paths)) for j in
                     range(len(destination_cells2destination_airports.keys()))], vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0,
                    name="phi")
    for airport in airports:
        neighbors = list(graph.neighbors(airport))
        m.addConstr(rho[airport] <= m1_vals[airport] * (1 - y[airport]))  # 12
        m.addConstr(
            y[airport] + gp.quicksum(w[(airport, neighbor)] for neighbor in neighbors) + chi[
                airport] == 1)  # 16
        m.addConstr(rho[airport] >= m1_vals[airport] * chi[airport])  # 15
        for neighbor in neighbors:
            edge = tuple(sorted((airport, neighbor)))
            m.addConstr(rho[airport] <= graph.edges[airport, neighbor]['weight'] + rho[neighbor])  # 13
            m.addConstr(
                rho[airport] >= graph.edges[airport, neighbor]['weight'] + rho[neighbor] - m2_vals[edge] * (
                        1 - w[(airport, neighbor)]))  # 14

        neighbors = list(graph.neighbors(airport))
        min_dist_to_neighbor = min(graph.edges[airport, neighbor]['weight'] for neighbor in neighbors)
        m.addConstr(rho[airport] >= min(min_dist_to_neighbor, m1_vals[airport]) * (1 - y[airport]))  # 20

    for airport_i, airport_j in canonical_edges:
        m.addConstr(graph.edges[airport_i, airport_j]['weight'] + rho[airport_i] + rho[airport_j] <=
                    settings.aircraft_config.tau + m3_vals[airport_i, airport_j] * (
                            1 - z[airport_i, airport_j]))  # 4
        m.addConstr(z[airport_i, airport_j] <= 1 - chi[airport_i])  # 21
        m.addConstr(z[airport_i, airport_j] <= 1 - chi[airport_j])  # 22
        m4_ij = settings.aircraft_config.tau - graph.edges[airport_i, airport_j]['weight']
        m.addConstr(graph.edges[airport_i, airport_j]['weight'] + rho[airport_i] + rho[airport_j] +
                    m4_ij * z[airport_i, airport_j] >= settings.aircraft_config.tau)  # 24

    for id, path in enumerate(paths):
        path_edges = [tuple(sorted((path[i], path[i + 1]))) for i in range(len(path) - 1)]
        m.addConstr(psi[id] - gp.quicksum(z[edge] for edge in path_edges) >= 1 - len(path_edges))  # 23
        for edge in path_edges:
            m.addConstr(psi[id] <= z[edge])  # 5

    for id in range(len(population_cells_paths)):
        for destination_id, destination in enumerate(destination_cells2destination_airports):
            path_indices = [i for i, p in enumerate(paths) if
                            p in population_cells_paths[id] and p[-1] in destination_cells2destination_airports[
                                destination]]
            if path_indices:
                m.addConstr(phi[id, destination_id] <= gp.quicksum(psi[p_id] for p_id in path_indices))
            else:
                m.addConstr(phi[id, destination_id] == 0)
    m.update()

    return m, y, phi

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


def get_outputs_from_model(m):
    y_vars, psi_vars, _ = get_y_psi_phi_variables(m)

    charging_airports = [int(v.VarName[2:-1]) for v in y_vars if v.X == 1]
    active_path_indices = np.array([int(v.VarName[4:-1]) for v in psi_vars if v.X == 1])

    nObjectives = m.NumObj
    nSolutions = m.SolCount
    solutions = defaultdict(list)
    for s in range(nSolutions):
        m.params.SolutionNumber = s
        if nObjectives > 1:
            for o in range(nObjectives):
                m.params.ObjNumber = o
                solutions[s].append(m.ObjNVal)
        else:
            solutions[s].append(m.ObjVal)

    return charging_airports, active_path_indices, solutions

def get_y_psi_phi_variables(m):
    all_vars = m.getVars()
    y_vars = [v for v in all_vars if v.VarName.startswith('y[')]
    psi_vars = [v for v in all_vars if v.VarName.startswith('psi[')]
    phi_vars = [v for v in all_vars if v.VarName.startswith('phi[')]

    return y_vars, psi_vars, phi_vars