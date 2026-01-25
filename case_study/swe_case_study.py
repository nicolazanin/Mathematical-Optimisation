import sys
import numpy as np
from gurobipy import GRB
import pandas as pd
import logging
import time
import os
import ast
import urllib.request

from utils.init_dataset import get_activation_cost_airports, get_nodes_distances, get_population_cells_near_airports, \
    get_population_cells2airports_distances, get_destinations_airports_info, get_nodes_distances_alt
from utils.preprocessing import (get_threshold_graph, get_attractive_paths_from_rft, get_all_paths_to_destinations,
                                 get_population_cells_paths, get_population_cells_too_close_to_destination_cells,
                                 get_attractive_paths, get_attractive_graph,
                                 get_airports_too_close_to_destination_cells)
from model.utils_model import get_outputs_from_model, get_model_variables
from model.eanc_reg_model import solve_eacn_model
from utils.case_study_utils import (get_airports, get_population_cells, get_population_cells_centroids, plot_case_study)
from utils.settings import settings, setup_logging

tic = time.time()

setup_logging(log_prefix="EACN-REG", print_file=settings.print_logs)
_logger = logging.getLogger(__name__)
_logger.setLevel(settings.logging_lvl)

_logger.info("-------------- EACN-REG SWE case study starting --------------")

settings = settings.from_yaml("case_study/config_case_study.yml")
url_airports_file = "https://github.com/nicolazanin/Mathematical-Optimisation/releases/latest/download/swe_airports.csv"
url_pop_density_file = "https://github.com/nicolazanin/Mathematical-Optimisation/releases/latest/download/swe_pd_2019_1km_ASCII_XYZ.csv"
airports_file = "swe_airports.csv"
pop_density_file = "swe_pd_2019_1km_ASCII_XYZ.csv"

if not os.path.exists(pop_density_file):
    _logger.info("Downloading latest population density file")
    urllib.request.urlretrieve(url_pop_density_file, pop_density_file)

if not os.path.exists(airports_file):
    _logger.info("Downloading latest airports file")
    urllib.request.urlretrieve(url_airports_file, airports_file)

_logger.info("-------------- Initialize the population grid dataset --------------")
df_population = pd.read_csv(filepath_or_buffer=pop_density_file)  # https://hub.worldpop.org/geodata/summary?id=44035
_logger.info("Population grid dataset initialized from {}".format(pop_density_file))
population_cells = get_population_cells(df_population=df_population, cell_area=settings.population_config.cell_area)
population_cells_centroids = get_population_cells_centroids(population_cells=population_cells)
population_coords = np.array([(pop_cell_centroid[3], pop_cell_centroid[4]) for pop_cell_centroid in
                              population_cells_centroids.values.tolist()])
population_density = np.array([pop_cell_centroid[2] for pop_cell_centroid in
                               population_cells_centroids.values.tolist()])

_logger.info("-------------- Initialize the airports dataset --------------")
_logger.info("Airports dataset initialized from {}".format(airports_file))
airports = get_airports(airports_file=airports_file, only_active=False)
activation_costs = get_activation_cost_airports(num_airports=np.size(airports),
                                                max_cost=settings.airports_config.max_cost,
                                                min_cost=settings.airports_config.min_cost)
airports_coords = np.array([(airport.x, airport.y) for airport in airports])
airports_distances_alt = get_nodes_distances_alt(nodes_coords=airports_coords, res=settings.paths_config.res)

_logger.info("-------------- Define Destination Airport/s --------------")
max_ground_distance = settings.ground_access_config.avg_speed * settings.ground_access_config.max_time / 60
population_cells_near_airports = get_population_cells_near_airports(airports_coords=airports_coords,
                                                                    population_coords=population_coords,
                                                                    max_ground_distance=max_ground_distance)
population_cells2airport_distances = get_population_cells2airports_distances(population_coords=population_coords,
                                                                             airports_coords=airports_coords)

destinations_airports_info = get_destinations_airports_info(
    destination_cells=settings.population_config.destination_cells,
    population_airport_distances=population_cells2airport_distances,
    max_ground_distance=max_ground_distance)
destination_airports = np.array([destination_airport_prop[1] for destination_airport_prop in
                                 destinations_airports_info if destination_airport_prop[1] is not None])

_logger.info("------------- Pre-Processing --------------")
# Create two graph to identify the edges above and below the distance threshold tau (single charge range)
airports_graph_below_tau_alt = get_threshold_graph(distances=airports_distances_alt,
                                                   tau=settings.aircraft_config.tau)

_logger.info("-------------- Define Paths --------------")
all_paths = get_all_paths_to_destinations(graph=airports_graph_below_tau_alt,
                                          destination_airports=destination_airports,
                                          max_path_edges=settings.paths_config.max_edges)
attractive_paths_from_rft = get_attractive_paths_from_rft(paths=all_paths,
                                                          distances=airports_distances_alt,
                                                          routing_factor_thr=settings.paths_config.routing_factor_thr)

min_distance_to_destination_cells = (settings.ground_access_config.avg_speed *
                                     settings.paths_config.min_ground_travel_time_to_destination_cell)
population_cells_too_close_to_destination_cells = get_population_cells_too_close_to_destination_cells(
    population_coords=population_coords,
    destination_cells=settings.population_config.destination_cells,
    min_distance_to_destination_cells=min_distance_to_destination_cells)
airports_too_close_to_destination_cells = (
    get_airports_too_close_to_destination_cells(airports_coords=airports_coords,
                                                population_coords=population_coords,
                                                destination_cells=settings.population_config.destination_cells,
                                                min_distance_to_destination_cells=min_distance_to_destination_cells))
population_cells_paths = (
    get_population_cells_paths(population_coords=population_coords,
                               paths=attractive_paths_from_rft,
                               distances=airports_distances_alt,
                               population_cells_near_airports=population_cells_near_airports,
                               destinations_airports_info=destinations_airports_info,
                               population_cells2airport_distances=population_cells2airport_distances,
                               population_cells_too_close_to_destination_cells=population_cells_too_close_to_destination_cells,
                               airports_too_close_to_destination_cells=airports_too_close_to_destination_cells,
                               ground_speed=settings.ground_access_config.avg_speed,
                               air_speed=settings.aircraft_config.cruise_speed,
                               max_total_time=settings.paths_config.max_total_time_travel))
attractive_paths = get_attractive_paths(population_cells_paths=population_cells_paths)
attractive_graph = get_attractive_graph(distances=airports_distances_alt, attractive_paths=attractive_paths)

_logger.info("-------------- MILP Optimization --------------")
m, time_exec = solve_eacn_model(population_density=population_density,
                                attractive_paths=attractive_paths,
                                activation_costs=activation_costs,
                                attractive_graph=attractive_graph,
                                population_cells_paths=population_cells_paths,
                                destinations_airports_info=destinations_airports_info,
                                tau=settings.aircraft_config.tau,
                                mu_1=settings.model_config.mu_1,
                                mu_2=settings.model_config.mu_2,
                                mip_gap=settings.model_config.mip_gap,
                                epsilon=settings.model_config.epsilon,
                                charging_bases_lim=settings.airports_config.charging_bases_lim,
                                lexicographic=settings.model_config.lexicographic,
                                ks=settings.heuristic_config.enable,
                                initial_kernel_size=settings.heuristic_config.initial_kernel_size,
                                buckets_size=settings.heuristic_config.buckets_size,
                                iterations=settings.heuristic_config.iterations,
                                max_run_time=settings.model_config.max_run_time)
charging_airports = []
active_path_indices = []
population_covered = [int(cell) for cells in population_cells_too_close_to_destination_cells.values() for cell in cells]
if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
    charging_airports, population_from_dest, active_path_indices, bound = get_outputs_from_model(m)
    population_covered = population_covered + population_from_dest
    _logger.info("Charging airports: {} ({})".format(str(charging_airports), len(charging_airports)))
    _logger.info("Population covered: {} ({})".format(str(population_covered), len(population_covered)))
else:
    _logger.info("No solution was found. Status:".format(m.Status))

if not settings.show_plot:
    _logger.info("Plot skipped")
else:
    _logger.info("-------------- Plot --------------")
    edges = []
    y, psi, phi, rho, chi, z, w = get_model_variables(m)
    for k, value in w.items():
        if value > 0.5:
            edges.append(ast.literal_eval(k[1:]))
    for k, value in z.items():
        if value > 0.5:
            if ast.literal_eval(k[1:])[0] in charging_airports and ast.literal_eval(k[1:])[1] in charging_airports:
                edges.append(ast.literal_eval(k[1:]))
    edges = list({tuple(sorted(x)) for x in edges})
    edges = [list(x) for x in edges]
    plot_name = "test_{}_{}".format(settings.airports_config.charging_bases_lim,settings.aircraft_config.tau)
    plot_case_study(population_cells=population_cells,
                    population_cells_centroids=population_cells_centroids,
                    airports=airports,
                    destination_airports=destination_airports,
                    destination_cells=settings.population_config.destination_cells,
                    edges=edges,
                    charging_airports=charging_airports,
                    population_covered=population_covered,
                    plot_name=plot_name,
                    save_plot=settings.save_plot)
