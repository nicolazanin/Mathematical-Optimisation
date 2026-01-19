import numpy as np
import time
import logging
from gurobipy import GRB
import networkx as nx

from utils.settings import settings
from model.utils_model import (get_initial_kernel, get_buckets, get_outputs_from_model, model, get_y_psi_phi_variables)

_logger = logging.getLogger(__name__)


def solve_eacn_model(population_density: np.ndarray, activation_costs: np.ndarray, attractive_paths: np.ndarray,
                     attractive_graph: nx.Graph, population_cells_paths: dict, destinations_airports_info: list,
                     tau: int, mu_1: float, mu_2: float, mip_gap: float, epsilon: int, lexicographic: bool,
                     ks: bool, initial_kernel_size: int, buckets_size: int, iterations: int,
                     max_run_time: int) -> tuple:
    """
    Solves the EACN model using Gurobi with the possibility to use the kernel search heuristic.

    Args:
        population_density (np.ndarray): A NumPy array of population density values (integers) with the same length as
            `population_coords`.
        activation_costs (np.ndarray): A NumPy array of activation costs for each airport (based on 'min_cost' and
            'max_cost').
        attractive_paths (np.ndarray): A NumPy array of attractive paths (each path is a list of node IDs).
        attractive_graph (nx.Graph): A NetworkX graph containing the filtered edges from the attractive paths.
        population_cells_paths (dict): A dictionary mapping each population cell index to a list of paths (each path is
            a list of node IDs) starting from an airport near that population cell.
        destinations_airports_info (list) : Each tuple in the list contains: (destination_cell_idx, closest_airport_idx,
            distance)
        tau (int): Maximum travel range on a single charge.
        mu_1 (float): Weight of first objective function.
        mu_2 (float): Weight of second objective function.
        mip_gap (float): MIP gap termination condition value.
        epsilon (int): Small positive number to define big-M parameters.
        lexicographic (bool): True if the model combines the two objective functions in a lexicographic order.
        ks (bool): True if the kernel search heuristic is enabled.
        initial_kernel_size (int): # Kernel search heuristic initial kernel size.
        buckets_size (int): Kernel search heuristic buckets size.
        iterations (int): Kernel search heuristic total iterations.
        max_run_time (int): The maximum run time in seconds.

    Returns:
        tuple: A tuple containing the model and the optimization solution time
    """
    m = None
    start_time = time.time()

    dest_airport_info = {dest_cell: airport_idx for dest_cell, airport_idx, _ in destinations_airports_info}
    attractive_airports = list(attractive_graph.nodes())

    if ks == True and lexicographic == True:
        _logger.warning("Kernel search heuristic is enabled but incompatible lexicographic order is selected "
                        "(blending approach is used)")  # blending approach is the linear combination of obt functions

    if ks:
        kernel = get_initial_kernel(population_cells_paths=population_cells_paths,
                                    initial_kernel_size=initial_kernel_size)
        best_obj_val = 0
        _logger.info("-------------- EACN-REG kernel search starting --------------")
        for iteration in range(iterations):
            buckets = get_buckets(airports=attractive_airports, kernel=kernel, bucket_size=buckets_size)
            if (time.time() - start_time) < max_run_time:
                for bucket_id, bucket in buckets.items():
                    _logger.info("-------------- Kernel search {} iteration, {} bucket--------------".
                                 format(str(iteration + 1), str(bucket_id + 1)))
                    candidates_airports = kernel + bucket
                    not_candidates_airports = list(set(attractive_airports) - set(candidates_airports))
                    m, y_vars, phi_vars = model(airports=attractive_airports, paths=attractive_paths,
                                                    graph=attractive_graph,
                                                    population_cells_paths=population_cells_paths,
                                                    destinations_airports_info=destinations_airports_info, tau=tau,
                                                    mip_gap=mip_gap, epsilon=epsilon, max_run_time=max_run_time)
                    population_covered = np.array([population_density[idx] * phi_vars[idx, dest_cell]
                                                   for idx in population_cells_paths for dest_cell in
                                                   dest_airport_info.keys()]).sum()
                    installation_cost = np.array([activation_costs[i] * y_vars[i] for i in attractive_airports]).sum()
                    objective_func = (settings.model_config.mu_1 * population_covered -
                                      settings.model_config.mu_2 * installation_cost)
                    m.setObjective(objective_func, GRB.MAXIMIZE)
                    m.addConstr(objective_func >= best_obj_val)
                    for not_candidates_airport in not_candidates_airports:
                        m.addConstr(y_vars[not_candidates_airport] == 0)
                    m.write("EACN_REG_model.lp")
                    m.optimize()
                    if m.Status == GRB.OPTIMAL:
                        charging_airports, active_path_indices, solutions = get_outputs_from_model(m)
                        kernel = kernel + [charging_airport for charging_airport in charging_airports
                                           if charging_airport not in kernel]
                        best_obj_val = solutions[0][0]
                    else:  # TIME_LIMIT, GRB.INFEASIBLE, GRB.UNBOUNDED
                        pass
                        # return m, time_limit
    else:
        m, y_vars, phi_vars = model(airports=attractive_airports, paths=attractive_paths, graph=attractive_graph,
                                    population_cells_paths=population_cells_paths,
                                    destinations_airports_info=destinations_airports_info, tau=tau, mip_gap=mip_gap,
                                    epsilon=epsilon, max_run_time=max_run_time)

        population_covered = np.array([population_density[idx] * phi_vars[idx, dest_cell]
                                       for idx in population_cells_paths for dest_cell in
                                       dest_airport_info.keys()]).sum()
        installation_cost = np.array([activation_costs[i] * y_vars[i] for i in attractive_airports]).sum()

        if lexicographic:
            m.setObjective(mu_1 * population_covered, GRB.MAXIMIZE)
            m.optimize()
            best_obj_val = m.ObjVal
            m.setObjective(mu_2 * installation_cost, GRB.MINIMIZE)
            m.addConstr(population_covered >= best_obj_val)
            m.optimize()
        else:
            objective_func = (mu_1 * population_covered - mu_2 * installation_cost)
            m.setObjective(objective_func, GRB.MAXIMIZE)
            m.optimize()

        m.write("EACN_REG_model.lp")

    return m, time.time() - start_time
