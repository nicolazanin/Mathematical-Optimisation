import numpy as np
from gurobipy import GRB
import logging
import time

from utils.init_dataset import (cells_generation, nodes_generation, get_population_cells_near_airports,
                                get_pop_density, get_nodes_distances, get_grid_dimensions,
                                get_activation_cost_airports, get_destinations_airports_info,
                                get_population_cells2airports_distances)
from utils.preprocessing import (get_threshold_graph, get_attractive_paths_from_rft, get_all_paths_to_destinations,
                                 get_population_cells_paths, get_population_cells_too_close_to_destination_cells,
                                 get_attractive_paths, get_attractive_graph)
from model.utils_model import get_outputs_from_model
from utils.plot import plot_dataset
from model.eanc_reg_model import solve_eacn_model
from utils.settings import settings, setup_logging

# settings = settings.from_yaml("config_scalability.yml")
# settings.random_seed = 0
# settings.airports_config.num = 50
# settings.aircraft_config.tau = 400
# settings.population_config.cells_x = 10
# settings.population_config.cells_y = 10
# settings.population_config.destination_cells = [int(
#     str(np.random.randint(settings.population_config.cells_y * 0.2, settings.population_config.cells_y * 0.8)) +
#     str(np.random.randint(settings.population_config.cells_x * 0.2, settings.population_config.cells_x * 0.8)))]
tic = time.time()

setup_logging(log_prefix="EACN-REG", print_file=settings.print_logs)
_logger = logging.getLogger(__name__)
_logger.setLevel(settings.logging_lvl)

_logger.info("-------------- EACN-REG test starting --------------")
_logger.info("-------------- N: {} --------------".format(settings.airports_config.num))
_logger.info("-------------- K: {} --------------".format(settings.population_config.cells_x *
                                                          settings.population_config.cells_y))
_logger.info("-------------- tau: {} --------------".format(settings.aircraft_config.tau))
_logger.info("-------------- routing_factor: {} --------------".format(settings.paths_config.routing_factor_thr))

np.random.seed(settings.random_seed)

_logger.info("-------------- Initialize the population grid dataset --------------")
population_coords = cells_generation(num_cells_x=settings.population_config.cells_x,
                                     num_cells_y=settings.population_config.cells_y,
                                     cell_area=settings.population_config.cell_area)
population_density = get_pop_density(population_coords=population_coords,
                                     min_density=settings.population_config.min_density,
                                     max_density=settings.population_config.max_density,
                                     high_population_cells=settings.population_config.high_population_cells)

_logger.info("-------------- Initialize the airports dataset --------------")
total_width_pop_area, total_height_pop_area = get_grid_dimensions(num_cells_x=settings.population_config.cells_x,
                                                                  num_cells_y=settings.population_config.cells_y,
                                                                  cell_area=settings.population_config.cell_area)
airports_coords = nodes_generation(num_nodes=settings.airports_config.num,
                                   total_width=total_width_pop_area,
                                   total_height=total_height_pop_area,
                                   additional_nodes=settings.airports_config.additional_airport_coords,
                                   min_distance_km=settings.airports_config.min_distance)
activation_costs = get_activation_cost_airports(num_airports=np.size(airports_coords),
                                                max_cost=settings.airports_config.max_cost,
                                                min_cost=settings.airports_config.min_cost)
airports_distances = get_nodes_distances(nodes_coords=airports_coords)

_logger.info("-------------- Define Destination Airport/s --------------")
max_ground_distance = settings.ground_access_config.avg_speed * settings.ground_access_config.max_time / 60
population_cells_near_airports = get_population_cells_near_airports(airports_coords=airports_coords,
                                                                    population_coords=population_coords,
                                                                    max_ground_distance=max_ground_distance)

population_cells2airport_distances = get_population_cells2airports_distances(population_coords=population_coords,
                                                                             airports_coords=airports_coords)

destinations_airports_info = get_destinations_airports_info(
    destination_cells=settings.population_config.destination_cells,
    population_airport_distances=population_cells2airport_distances, max_ground_distance=max_ground_distance)
destination_airports = np.array([destination_airport_prop[1] for destination_airport_prop in
                                 destinations_airports_info if destination_airport_prop[1] is not None])

_logger.info("------------- Pre-Processing --------------")
# Create two graph to identify the edges above and below the distance threshold tau (single charge range)
airports_graph_below_tau = get_threshold_graph(distances=airports_distances,
                                               tau=settings.aircraft_config.tau)
airports_graph_above_tau = get_threshold_graph(distances=airports_distances,
                                               tau=settings.aircraft_config.tau, mode='above')

_logger.info("-------------- Define Paths --------------")
all_paths = get_all_paths_to_destinations(graph=airports_graph_below_tau,
                                          destination_airports=destination_airports,
                                          max_path_edges=settings.paths_config.max_edges)
attractive_paths_from_rft = get_attractive_paths_from_rft(paths=all_paths, distances=airports_distances,
                                                          routing_factor_thr=settings.paths_config.routing_factor_thr)

min_distance_to_destination_cells = (settings.ground_access_config.avg_speed *
                                     settings.paths_config.min_ground_travel_time_to_destination_cell)
population_cells_too_close_to_destination_cells = get_population_cells_too_close_to_destination_cells(
    population_coords=population_coords, destination_cells=settings.population_config.destination_cells,
    min_distance=min_distance_to_destination_cells)
population_cells_paths = (
    get_population_cells_paths(population_coords=population_coords,
                               paths=attractive_paths_from_rft, distances=airports_distances,
                               population_cells_near_airports=population_cells_near_airports,
                               destinations_airports_info=destinations_airports_info,
                               population_cells2airport_distances=population_cells2airport_distances,
                               population_cells_too_close_to_destination_cells=population_cells_too_close_to_destination_cells,
                               ground_speed=settings.ground_access_config.avg_speed,
                               air_speed=settings.aircraft_config.cruise_speed,
                               max_total_time=settings.paths_config.max_total_time_travel))
attractive_paths = get_attractive_paths(population_cells_paths=population_cells_paths)
attractive_graph = get_attractive_graph(distances=airports_distances, attractive_paths=attractive_paths)

_logger.info("-------------- MILP Optimization --------------")
m, time_exec = solve_eacn_model(population_density=population_density, attractive_paths=attractive_paths,
                                activation_costs=activation_costs, attractive_graph=attractive_graph,
                                population_cells_paths=population_cells_paths,
                                destinations_airports_info=destinations_airports_info,
                                tau=settings.aircraft_config.tau, mu_1=settings.model_config.mu_1,
                                mu_2=settings.model_config.mu_2, mip_gap=settings.model_config.mip_gap,
                                epsilon=settings.model_config.epsilon,
                                lexicographic=settings.model_config.lexicographic,
                                ks=settings.heuristic_config.enable,
                                initial_kernel_size=settings.heuristic_config.initial_kernel_size,
                                buckets_size=settings.heuristic_config.buckets_size,
                                iterations=settings.heuristic_config.iterations,
                                max_run_time=settings.model_config.max_run_time)
charging_airports = []
active_path_indices = []
if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
    charging_airports, population_covered, active_path_indices, bound = get_outputs_from_model(m)
    _logger.info("Charging airports: {} ({})".format(str(charging_airports), len(charging_airports)))
    _logger.info("Population covered: {} ({})".format(str(population_covered), len(population_covered)))
else:
    _logger.info("No solution was found. Status:".format(m.Status))

if not settings.show_plot:
    _logger.info("Plot skipped")
else:
    _logger.info("-------------- Plot --------------")
    plot_name = "test_{}_{}_{}_".format(settings.airports_config.num,
                                        settings.population_config.cells_x * settings.population_config.cells_y,
                                        settings.aircraft_config.tau)
    plot_dataset(population_coords=population_coords, population_density=population_density,
                 airports_coords=airports_coords, airport_distances=airports_distances,
                 graph_below_tau=airports_graph_below_tau, graph_above_tau=airports_graph_above_tau,
                 destination_airports=destination_airports,
                 destination_cells=settings.population_config.destination_cells,
                 max_ground_distance=max_ground_distance, all_paths=all_paths,
                 attractive_paths=attractive_paths,
                 population_cells_paths=population_cells_paths, charging_airports=charging_airports,
                 active_path_indices=active_path_indices, plot_name=plot_name,
                 simple_plot_enable=settings.simple_plot_enable, save_plot=settings.save_plot)

_logger.info("Total execution time for EACN-REG: {:.1f} minutes".format((time.time() - tic) / 60))
