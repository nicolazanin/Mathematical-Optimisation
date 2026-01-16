import numpy as np
import time
import logging
from gurobipy import GRB
from utils.settings import settings

from model.utils_model import (get_initial_kernel, get_buckets, get_outputs_from_model, model)

_logger = logging.getLogger(__name__)


def solve_eacn_model(population_density, activation_costs, attractive_paths, active_graph,
                     population_cells_paths, destination_cells, destination_airports, ks, objective_lex=0):
    m = None
    start_time = time.time()
    cols = None
    rows = None
    lb = None
    if ks:
        kernel = get_initial_kernel(population_cells_paths=population_cells_paths,
                                    initial_kernel_size=settings.heuristic_config.initial_kernel_size)
        active_airports = active_graph.nodes()
        best_obj_val = 0
        _logger.info("-------------- EACN-REG kernel search starting --------------")
        for iteration in range(settings.heuristic_config.iterations):
            buckets = get_buckets(active_airports, kernel, settings.heuristic_config.bucket_size)
            if (time.time() - start_time) < settings.heuristic_config.max_run_time:
                for bucket_id, bucket in buckets.items():
                    _logger.info("-------------- Kernel search {} iteration, {} bucket--------------".
                                 format(str(iteration + 1), str(bucket_id + 1)))
                    airports = kernel + bucket
                    graph = remove_nodes_from_graph(graph=active_graph, nodes_to_keep=airports, print_action=True)
                    paths = np.array(
                        [path for path in attractive_paths if all(node in airports for node in path)], dtype=object)
                    new_m, y_vars, phi_vars = model(airports, paths, graph, population_cells_paths,
                                                    destination_cells2destination_airports)
                    population_covered = np.array(
                        [population_density[id] * phi_vars[id, _] for id in range(len(population_cells_paths))
                         for _ in range(len(destination_cells2destination_airports.keys()))]).sum()
                    installation_cost = np.array([activation_costs[i] * y_vars[i] for i in airports]).sum()
                    objective_func = (settings.model_config.mu_1 * population_covered -
                                      settings.model_config.mu_2 * installation_cost)
                    new_m.setObjective(objective_func, GRB.MAXIMIZE)
                    new_m.addConstr(objective_func >= best_obj_val)
                    new_m.setParam('TimeLimit', settings.heuristic_config.max_run_time)
                    new_m.write("EACN_REG_model.lp")
                    new_m.optimize()
                    if m.Status == GRB.OPTIMAL:
                        charging_airports, active_path_indices, solutions = get_outputs_from_model(m)
                        kernel = kernel + [charging_airport for charging_airport in charging_airports
                                           if charging_airport not in kernel]
                        best_obj_val = solutions[0][0]
                        m = new_m
                    else:  # TIME_LIMIT, GRB.INFEASIBLE, GRB.UNBOUNDED
                        pass
                        # return m, time_limit
    else:
        active_airports = active_graph.nodes()
        m, y_vars, phi_vars = model(active_airports, attractive_paths, active_graph, population_cells_paths,
                                    destination_cells, destination_airports)
        population_covered = np.array(
            [population_density[id] * phi_vars[id, _] for id in range(len(population_cells_paths))
             for _ in range(len(destination_cells))]).sum()
        installation_cost = np.array([activation_costs[i] * y_vars[i] for i in active_airports]).sum()
        if settings.model_config.lexicographic:
            m.setObjective(settings.model_config.mu_1 * population_covered, GRB.MAXIMIZE)
            m.setParam('TimeLimit', settings.heuristic_config.max_run_time)
            m.optimize()
            best_obj_val = m.ObjVal
            m.setObjective(settings.model_config.mu_2 * installation_cost, GRB.MINIMIZE)
            objective_func = (settings.model_config.mu_1 * population_covered)
            m.addConstr(objective_func >= best_obj_val)
            m.setParam('TimeLimit', settings.heuristic_config.max_run_time)
            m.optimize()
            cols = m.NumVars
            rows = m.NumConstrs
            lb = m.ObjBound
        else:
            objective_func = (settings.model_config.mu_1 * population_covered -
                              settings.model_config.mu_2 * installation_cost)
            m.setObjective(objective_func, GRB.MAXIMIZE)
            m.setParam('TimeLimit', settings.heuristic_config.max_run_time)
            m.optimize()

    return m, time.time() - start_time, cols, rows, lb
