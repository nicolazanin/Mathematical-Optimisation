import numpy as np
import gurobipy as gp
from gurobipy import GRB
from utils.settings import settings

from model.utils_model import calculate_tight_big_m, get_initial_kernel, get_buckets, get_outputs_from_model
from utils.preprocessing import remove_nodes_from_graph


def solve_eacn_model(population_density, attractive_paths, activation_costs, airports_graph_below_tau, active_airports,
                     population_cells_paths, destination_cells2destination_airports, ks):
    if ks:
        initial_kernel = get_initial_kernel(population_cells_paths=population_cells_paths,
                                            initial_kernel_size=settings.heuristic_config.initial_kernel_size)
        buckets = get_buckets(active_airports, initial_kernel, settings.heuristic_config.bucket_size)
        iterations = settings.heuristic_config.iterations
    else:
        buckets = {0: []}
        initial_kernel = active_airports
        iterations = 1

    kernel = initial_kernel
    best_obj_val = 0
    for iteration in range(iterations):
        for bucket_id, bucket in buckets.items():
            m = gp.Model("EACN_REG")
            m.setParam('LogToConsole', 0)
            airports = kernel + bucket
            graph = remove_nodes_from_graph(graph=airports_graph_below_tau, nodes_to_keep=airports, print_action=True)
            paths = np.array([path for path in attractive_paths if all(node in airports for node in path)],
                             dtype=object)

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
            phi = m.addVars([(i, j) for i in range(len(population_density)) for j in
                             range(len(destination_cells2destination_airports.keys()))], vtype=GRB.CONTINUOUS, lb=0.0,
                            ub=1.0,
                            name="phi")
            population_covered = np.array(
                [population_density[id] * phi[id, _] for id in range(len(population_density)) for _ in
                 range(len(destination_cells2destination_airports.keys()))]).sum()
            installation_cost = np.array(activation_costs[airports] * [y[i] for i in range(len(airports))]).sum()

            if ks:
                objective_func = settings.model_config.mu_1 * population_covered - settings.model_config.mu_2 * installation_cost
                m.setObjective(objective_func, GRB.MAXIMIZE)
                m.addConstr(objective_func >= best_obj_val)
            elif settings.model_config.lexicographic:
                m.setObjectiveN(population_covered, index=0, priority=2, weight=-1)
                m.setObjectiveN(installation_cost, index=1, priority=1, weight=1)
            else:
                objective_func = settings.model_config.mu_1 * population_covered - settings.model_config.mu_2 * installation_cost
                m.setObjective(objective_func, GRB.MAXIMIZE)

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

            for id, path in enumerate(attractive_paths):
                path_edges = [tuple(sorted((path[i], path[i + 1]))) for i in range(len(path) - 1)]
                m.addConstr(psi[id] - gp.quicksum(z[edge] for edge in path_edges) >= 1 - len(path_edges))  # 23
                for edge in path_edges:
                    m.addConstr(psi[id] <= z[edge])  # 5

            for id in range(len(population_density)):
                for destination_id, destination in enumerate(destination_cells2destination_airports):
                    path_indices = [i for i, p in enumerate(attractive_paths) if
                                    p in population_cells_paths[id] and p[-1] in destination_cells2destination_airports[
                                        destination]]
                    if path_indices:
                        m.addConstr(phi[id, destination_id] <= gp.quicksum(psi[p_id] for p_id in path_indices))
                    else:
                        m.addConstr(phi[id, destination_id] == 0)

            m.setParam('TimeLimit', 5000)
            m.optimize()
            #m.write("EACN_REG_model.lp")
            if m.Status in (GRB.OPTIMAL,) and m.SolCount > 0:
                charging_airports, active_path_indices, solutions = get_outputs_from_model(m)
                kernel = kernel + [charging_airports for charging_airport in charging_airports
                                   if charging_airport not in kernel]
                best_obj_val = solutions[0][0]
    return m
