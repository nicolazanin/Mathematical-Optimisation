import numpy as np
import gurobipy as gp
from gurobipy import GRB
from utilis.settings import settings


def calculate_tight_big_m(active_graph):
    m1_vals = {}
    for airport in list(active_graph.nodes):
        neighbors = list(active_graph.neighbors(airport))
        min_dist_to_neighbor = min(active_graph.edges[airport,neighbor]['weight'] for neighbor in neighbors)
        m1_vals[airport] = settings.aircraft_config.tau - min_dist_to_neighbor + settings.model_config.epsilon

    m2_vals = {}
    m3_vals = {}
    for i, j in active_graph.edges():
        edge = tuple(sorted((i, j)))

        neighbors_i = list(active_graph.neighbors(edge[0]))
        neighbors_j = list(active_graph.neighbors(edge[1]))

        min_dist_from_j = min(active_graph.edges[edge[1],neighbor]['weight'] for neighbor in neighbors_j)
        min_dist_from_i = min(active_graph.edges[edge[0],neighbor]['weight'] for neighbor in neighbors_i)

        m2_vals[edge] = (active_graph.edges[edge[0],edge[1]]['weight'] + settings.aircraft_config.tau - min_dist_from_j +
                         settings.model_config.epsilon)
        m3_vals[edge] = (active_graph.edges[edge[0],edge[1]]['weight'] + settings.aircraft_config.tau - min_dist_from_i -
                         min_dist_from_j + settings.model_config.epsilon * 2)

    return m1_vals, m2_vals, m3_vals


def solve_eacn_model(population_density, attractive_paths, activation_costs, active_graph, active_airports,
                     population_cells_paths, destination_cell2destination_airport, ks=False, kernel=None, bucket=None, best_obj_value=-float('inf')):
    if ks:
        candidate_airports = kernel.union(bucket)  # MODIFICA KS

    m1_vals, m2_vals, m3_vals = calculate_tight_big_m(active_graph)

    m = gp.Model("EACN_REG_Strengthened")
    m.setParam('LogToConsole', 0)
    y = m.addVars(len(active_airports), vtype=GRB.BINARY, name="y")
    rho = m.addVars(len(active_airports), vtype=GRB.CONTINUOUS, name="rho", lb=0.0)
    w = m.addVars([(i, j) for i in range(len(active_airports)) for j in active_graph.neighbors(i)],
                  vtype=GRB.BINARY, name="w")
    chi = m.addVars(len(active_airports), vtype=GRB.BINARY, name="chi")

    canonical_edges = sorted(list(set(tuple(sorted(edge)) for edge in active_graph.edges())))

    z = m.addVars(canonical_edges, vtype=GRB.BINARY, name="z")

    psi = m.addVars(len(attractive_paths), vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="psi")
    phi = m.addVars([(i, j) for i in range(len(population_density)) for j in
                     range(len(settings.population_config.destination_cells))], vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0,
                    name="phi")
    population_covered = np.array([population_density[id]*phi[id,_] for id in range(len(population_density)) for _ in range(len(settings.population_config.destination_cells))]).sum()
    installation_cost = np.array(activation_costs[active_airports]*[y[i] for i in range(len(active_airports))]).sum()
    objective_func = settings.model_config.mu_1 * population_covered - settings.model_config.mu_2 * installation_cost
    m.setObjective(objective_func, GRB.MAXIMIZE)

    if ks:
        m.addConstr(objective_func >= best_obj_value + 0.001, name="obj_improvement_constraint")  # MODIFICA KS

    for airport in active_airports:
        if ks:
            if airport not in candidate_airports:  # MODIFICA KS
                y[i].ub = 0
        neighbors = list(active_graph.neighbors(airport))
        m.addConstr(rho[airport] <= m1_vals[airport] * (1 - y[airport])) # 12
        m.addConstr(y[airport] + gp.quicksum(w[(airport, neighbor)] for neighbor in neighbors) + chi[airport] == 1) # 16
        m.addConstr(rho[airport] >= m1_vals[airport] * chi[airport]) # 15
        for neighbor in neighbors:
            edge = tuple(sorted((airport, neighbor)))
            m.addConstr(rho[airport] <= active_graph.edges[airport,neighbor]['weight'] + rho[neighbor]) # 13
            m.addConstr(rho[airport] >= active_graph.edges[airport,neighbor] ['weight']+ rho[neighbor] - m2_vals[edge] * (
                    1 - w[(airport, neighbor)])) # 14

        neighbors = list(active_graph.neighbors(airport))
        min_dist_to_neighbor = min(active_graph.edges[airport,neighbor]['weight'] for neighbor in neighbors)
        m.addConstr(rho[airport] >= min(min_dist_to_neighbor, m1_vals[airport]) * (1 - y[airport])) # 20

    for airport_i, airport_j in canonical_edges:
        m.addConstr(active_graph.edges[airport_i, airport_j]['weight'] + rho[airport_i] + rho[airport_j] <=
                    settings.aircraft_config.tau + m3_vals[airport_i, airport_j] * (1 - z[airport_i, airport_j])) # 4
        m.addConstr(z[airport_i, airport_j] <= 1 - chi[airport_i]) # 21
        m.addConstr(z[airport_i, airport_j] <= 1 - chi[airport_j]) # 22
        m4_ij = settings.aircraft_config.tau - active_graph.edges[airport_i, airport_j]['weight']
        m.addConstr(active_graph.edges[airport_i, airport_j]['weight'] + rho[airport_i] + rho[airport_j] +
                    m4_ij * z[airport_i, airport_j] >= settings.aircraft_config.tau) # 24

    for id, path in enumerate(attractive_paths):
        path_edges = [tuple(sorted((path[i], path[i + 1]))) for i in range(len(path) - 1)]
        m.addConstr(psi[id] - gp.quicksum(z[edge] for edge in path_edges) >= 1 - len(path_edges)) # 23
        for edge in path_edges:
            m.addConstr(psi[id] <= z[edge]) # 5

    for id in range(len(population_density)):
        for destination_id, destination in enumerate(destination_cell2destination_airport):
            s = []
            for path in population_cells_paths[id]:
                if path[-1] in destination_cell2destination_airport[destination]:
                    for j, _ in enumerate(attractive_paths):
                        if np.array_equal(_,path):
                            s.append(psi[j])
            m.addConstr(phi[id, destination_id] <= gp.quicksum(s))

    m.setParam('TimeLimit', 600)
    m.optimize()

    return m
