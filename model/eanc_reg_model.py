import numpy as np
import time
from gurobipy import GRB
from utils.settings import settings

from model.utils_model import (get_initial_kernel, get_buckets, get_outputs_from_model, model)
from utils.preprocessing import remove_nodes_from_graph


def solve_eacn_model(population_density, activation_costs, attractive_paths, active_graph,
                     population_cells_paths, destination_cells2destination_airports, ks):
    if ks:
        kernel = get_initial_kernel(population_cells_paths=population_cells_paths,
                                    initial_kernel_size=settings.heuristic_config.initial_kernel_size)
        active_airports = active_graph.nodes()
        start_time = time.time()
        best_obj_val = 0
        for iteration in range(settings.heuristic_config.iterations):
            buckets = get_buckets(active_airports, kernel, settings.heuristic_config.bucket_size)
            if (time.time() - start_time) < settings.heuristic_config.max_run_time:
                for bucket_id, bucket in buckets.items():
                    airports = kernel + bucket
                    graph = remove_nodes_from_graph(graph=active_graph, nodes_to_keep=airports, print_action=True)
                    paths = np.array(
                        [path for path in attractive_paths if all(node in airports for node in path)], dtype=object)
                    m, y_vars, phi_vars = model(airports, paths, graph, population_cells_paths,
                                                destination_cells2destination_airports)
                    population_covered = np.array(
                        [population_density[id] * phi_vars[id, _] for id in range(len(population_cells_paths))
                         for _ in range(len(destination_cells2destination_airports.keys()))]).sum()
                    installation_cost = np.array([activation_costs[i] * y_vars[i] for i in airports]).sum()
                    objective_func = (settings.model_config.mu_1 * population_covered -
                                      settings.model_config.mu_2 * installation_cost)
                    m.setObjective(objective_func, GRB.MAXIMIZE)
                    m.addConstr(objective_func >= best_obj_val)
                    m.setParam('TimeLimit', settings.heuristic_config.max_run_time)
                    m.optimize()
                    if m.Status == GRB.OPTIMAL:
                        charging_airports, active_path_indices, solutions = get_outputs_from_model(m)
                        kernel = kernel + [charging_airport for charging_airport in charging_airports
                                           if charging_airport not in kernel]
                        best_obj_val = solutions[0][0]
                    else:  # TIME_LIMIT, GRB.INFEASIBLE, GRB.UNBOUNDED
                        pass
    else:
        active_airports = active_graph.nodes()
        m, y_vars, phi_vars = model(active_airports, attractive_paths, active_graph, population_cells_paths,
                                    destination_cells2destination_airports)
        population_covered = np.array(
            [population_density[id] * phi_vars[id, _] for id in range(len(population_cells_paths))
             for _ in range(len(destination_cells2destination_airports.keys()))]).sum()
        installation_cost = np.array([activation_costs[i] * y_vars[i] for i in active_airports]).sum()
        if settings.model_config.lexicographic:
            m.setObjectiveN(population_covered, index=0, priority=2, weight=-settings.model_config.mu_1)
            m.setObjectiveN(installation_cost, index=1, priority=1, weight=settings.model_config.mu_2)
        else:
            objective_func = (settings.model_config.mu_1 * population_covered -
                              settings.model_config.mu_2 * installation_cost)
            m.setObjective(objective_func, GRB.MAXIMIZE)
        m.optimize()
    return m
