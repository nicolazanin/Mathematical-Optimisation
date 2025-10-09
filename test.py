import sys
import numpy as np
from gurobipy import GRB
import logging
import time

from utils.init_dataset import (cells_generation, nodes_generation, get_population_cells_near_airports,
                                get_pop_density, nodes_distances, grid_dimensions,
                                get_destination_cells2destination_airports, get_activation_cost_airports,
                                get_closest_airport_from_destination_cell)
from utils.preprocessing import (create_threshold_graph, get_attractive_paths, get_all_paths_to_destinations,
                                 get_population_cells_paths, get_active_airports, get_active_graph)
from utils.plot import plot_dataset
from utils.eanc_reg_model import solve_eacn_model
from utils.settings import settings, setup_logging

tic = time.time()

setup_logging(log_prefix="EACN-REG", print_file=settings.print_logs)
_logger = logging.getLogger(__name__)
_logger.setLevel(settings.logging_lvl)

_logger.info("-------------- EACN-REG test starting --------------")

_logger.info("-------------- Initialize the population grid dataset --------------")
population_coords = cells_generation(num_cells_x=settings.population_config.cells_x,
                                     num_cells_y=settings.population_config.cells_y,
                                     cell_area=settings.population_config.cell_area)
population_density = get_pop_density(population_coords=population_coords,
                                     min_density=settings.population_config.min_density,
                                     max_density=settings.population_config.max_density)

_logger.info("-------------- Initialize the airports dataset --------------")
total_width_pop_area, total_height_pop_area = grid_dimensions(num_cells_x=settings.population_config.cells_x,
                                                              num_cells_y=settings.population_config.cells_y,
                                                              cell_area=settings.population_config.cell_area)
airports_coords = nodes_generation(num_nodes=settings.airports_config.num,
                                   total_width=total_width_pop_area,
                                   total_height=total_height_pop_area,
                                   min_distance_km=settings.airports_config.min_distance)
activation_costs = get_activation_cost_airports(num_airports=settings.airports_config.num,
                                                max_cost=settings.airports_config.max_cost,
                                                min_cost=settings.airports_config.min_cost)
airports_distances = nodes_distances(nodes_coords=airports_coords)

_logger.info("-------------- Define Destination Airport/s --------------")
max_ground_distance = settings.ground_access_config.avg_speed * settings.ground_access_config.max_time / 60
population_cells_near_airports = get_population_cells_near_airports(airports_coords=airports_coords,
                                                                    population_coords=population_coords,
                                                                    max_ground_distance=max_ground_distance)
if settings.model_config.closest_airport_to_destination_cell:
    destination_cells2destination_airports = {
        settings.population_config.destination_cells[0]: [get_closest_airport_from_destination_cell(
            airports_coords=airports_coords, population_coords=population_coords,
            destination_cell=settings.population_config.destination_cells[0])]}
    _logger.info("Destination airport is chosen based on the minimum distance from the first destination cell.")
    destination_airports = [destination_airport for airports in destination_cells2destination_airports.values()
                            for destination_airport in airports]
    _logger.info("For destination cells {}, the selected destination airports is: {}".
                 format(settings.population_config.destination_cells[0], destination_airports))

else:
    destination_cells2destination_airports = get_destination_cells2destination_airports(
        destination_cells=settings.population_config.destination_cells,
        population_cells_near_airports=population_cells_near_airports)
    _logger.info("Destination airports based on the maximum ground distance from the destination cells")
    destination_airports = [destination_airport for airports in destination_cells2destination_airports.values()
                            for destination_airport in airports]
    _logger.info("For destination cells {}, the selected destination airports is/are: {}".
                 format(settings.population_config.destination_cells, destination_airports))

if len(destination_airports) == 0:
    _logger.error("Empty destination airports list.")
    sys.exit()

destination_airports = np.array(destination_airports)

_logger.info("------------- Pre-Processing --------------")
# Create two graph to identify the edges above and below the distance threshold tau (single charge range)
airports_graph_below_tau = create_threshold_graph(distances=airports_distances,
                                                  tau=settings.aircraft_config.tau)
airports_graph_above_tau = create_threshold_graph(distances=airports_distances,
                                                  tau=settings.aircraft_config.tau, mode='above')

_logger.info("-------------- Define Paths --------------")
all_paths = get_all_paths_to_destinations(graph=airports_graph_below_tau,
                                          destination_airports=destination_airports,
                                          max_path_edges=settings.paths_config.max_edges)
attractive_paths = get_attractive_paths(paths=all_paths,
                                        distances=airports_distances,
                                        routing_factor_thr=settings.paths_config.routing_factor_thr)
active_airports = get_active_airports(attractive_paths=attractive_paths)
active_graph = get_active_graph(attractive_paths=attractive_paths, airports_distances=airports_distances)

population_cells_paths = get_population_cells_paths(population_coords=population_coords,
                                                    paths=attractive_paths,
                                                    population_cells_near_airports=population_cells_near_airports)

_logger.info("-------------- MILP Optimization --------------")
m = solve_eacn_model(population_density=population_density, attractive_paths=attractive_paths,
                     activation_costs=activation_costs, active_graph=active_graph, active_airports=active_airports,
                     population_cells_paths=population_cells_paths,
                     destination_cells2destination_airports=destination_cells2destination_airports)
charging_airports = []
active_path_indices = []
if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:

    all_vars = m.getVars()
    y_vars = [v for v in all_vars if v.VarName.startswith('y[')]
    psi_vars = [v for v in all_vars if v.VarName.startswith('psi[')]

    charging_airports = [int(v.VarName[2:-1]) for v in y_vars if v.X == 1]
    active_path_indices = np.array([int(v.VarName[4:-1]) for v in psi_vars if v.X == 1])
    _logger.info("Charging airports: {}".format(str(charging_airports)))
    _logger.info("Active paths: {}".format(str(active_path_indices)))

    nObjectives = m.NumObj
    nSolutions = m.SolCount
    for s in range(nSolutions):
        m.params.SolutionNumber = s
        obj_vals = []
        if settings.model_config.lexicographic:
            for o in range(nObjectives):
                m.params.ObjNumber = o
                obj_vals.append(m.ObjNVal)
        else:
            obj_vals = [m.ObjVal]
        _logger.info("Solutions {}: {} ".format(s, obj_vals))

    if active_path_indices.size == 0:
        _logger.info("No feasible path was found in the solution.")

else:
    _logger.info("No feasible solution was found. Status:", m.Status)

if not settings.plot:
    _logger.info("Plot skipped")
else:
    _logger.info("-------------- Plot --------------")
    plot_dataset(population_coords=population_coords, population_density=population_density,
                 airports_coords=airports_coords, airport_distances=airports_distances,
                 graph_below_tau=airports_graph_below_tau, graph_above_tau=airports_graph_above_tau,
                 destination_airports=destination_airports,
                 destination_cells=list(destination_cells2destination_airports.keys()),
                 max_ground_distance=max_ground_distance, all_paths=all_paths,
                 attractive_paths=attractive_paths,
                 population_cells_near_airports=population_cells_near_airports, charging_airports=charging_airports,
                 active_path_indices=active_path_indices,
                 closest_airport_to_destination_cell= settings.model_config.closest_airport_to_destination_cell)

_logger.info("Total execution time for EACN-REG: {:.1f} minutes".format((time.time() - tic) / 60))
