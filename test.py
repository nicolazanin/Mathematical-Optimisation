import sys
import pandas as pd
from gurobipy import GRB
import logging
import time

from utilis.init_dataset import (cells_generation, nodes_generation, get_population_cells_near_airports,
                                 get_pop_density, nodes_distances, grid_dimensions, get_destination_airports,
                                 get_activation_cost_airports)
from utilis.preprocessing import (create_threshold_graph, get_attractive_paths, get_all_paths_to_destinations,
                                  get_population_cells_paths, get_active_airports, get_active_graph)
from utilis.plot import plot_dataset
from utilis.eanc_reg_model import solve_eacn_model
from utilis.settings import settings, setup_logging

tic = time.time()

setup_logging(log_prefix="EACN-REG", print_file=settings.print_logs)
_logger = logging.getLogger(__name__)
_logger.setLevel(settings.logging_lvl)

_logger.info("-------------- EACN-REG starting --------------")

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

_logger.info("-------------- Define simple paths --------------")
max_ground_distance = settings.ground_access_config.avg_speed * settings.ground_access_config.max_time / 60
population_cells_near_airports = get_population_cells_near_airports(airports_coords=airports_coords,
                                                                    population_coords=population_coords,
                                                                    max_ground_distance=max_ground_distance)
destination_airports, destination_cell2destination_airports = get_destination_airports(
    destination_cells=settings.population_config.destination_cells,
    population_cells_near_airports=population_cells_near_airports)

_logger.info("For destination cells {}, the selected destination airports are {} based on the maximum ground distance."
             .format(settings.population_config.destination_cells, destination_airports))

if len(destination_airports) == 0:
    _logger.error("Empty destination airports list.")
    sys.exit()

_logger.info("------------- Pre-Processing --------------")
# Create two graph to identify the edges above and below the distance threshold tau (single charge range)
airports_graph_below_tau = create_threshold_graph(distances=airports_distances,
                                                  tau=settings.aircraft_config.tau)
airports_graph_above_tau = create_threshold_graph(distances=airports_distances,
                                                  tau=settings.aircraft_config.tau, mode='above')
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
m = solve_eacn_model(population_density, attractive_paths, activation_costs, active_graph, active_airports,
                     population_cells_paths, destination_cell2destination_airports)
active_bases = []
if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:

    all_vars = m.getVars()
    y_vars = [v for v in all_vars if v.VarName.startswith('y[')]
    psi_vars = [v for v in all_vars if v.VarName.startswith('psi[')]

    active_bases = [int(v.VarName[2:-1]) for v in y_vars if v.X > 0.5]
    _logger.info("Basi di Ricarica Attive ({}): {}".format(len(active_bases), str(active_bases)))

    _logger.info(f"Valore Funzione Obiettivo: {m.ObjVal:,.2f}")
    _logger.info(f"MIP Gap: {m.MIPGap:.4%}")

else:
    _logger.info("Nessuna soluzione trovata. Stato Gurobi:", m.Status)

if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:

    all_vars = m.getVars()
    psi_vars = [v for v in all_vars if v.VarName.startswith('psi[')]

    active_path_indices = [int(v.VarName[4:-1]) for v in psi_vars if v.X > 0.5]
    solution_paths = [attractive_paths[i] for i in active_path_indices]

    if not solution_paths:
        _logger.info("Nessun percorso è risultato fattibile nella soluzione trovata.")
else:
    _logger.info("Nessuna soluzione valida trovata. Impossibile visualizzare i percorsi.")

if settings.airports_config.num > 20:
    _logger.info("Plot skipped due to the large dataset")
else:
    _logger.info("-------------- Plot --------------")
    plot_dataset(population_coords=population_coords, population_density=population_density,
                 airports_coords=airports_coords, airport_distances=airports_distances,
                 graph_below_tau=airports_graph_below_tau, graph_above_tau=airports_graph_above_tau,
                 destination_airports=destination_airports,
                 destination_cells=settings.population_config.destination_cells,
                 max_ground_distance=max_ground_distance, all_paths=all_paths,
                 attractive_paths=attractive_paths,
                 population_cells_near_airports=population_cells_near_airports, charging_airports=active_bases)

_logger.info("Total execution time for EACN-REG: {:.1f} minutes".format((time.time() - tic) / 60))