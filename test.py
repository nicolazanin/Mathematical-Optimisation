import pandas as pd
from gurobipy import GRB
import logging

from utilis.init_dataset import (cells_generation, nodes_generation, get_pop_cells_near_airports, get_pop_density,
                                 nodes_distances, grid_dimensions)
from utilis.preprocessing import create_threshold_graph, get_attractive_paths
from utilis.init_model import get_all_simple_path_to_destinations, get_pop_paths
from utilis.plot import plot_dataset
from utilis.eanc_reg_model import solve_eacn_model
from utilis.settings import settings, setup_logging

setup_logging(log_prefix="EACN-REG", print_file=settings.print_logs)
_logger = logging.getLogger(__name__)
_logger.setLevel(settings.logging_lvl)

_logger.info("-------------- EACN-REG framework starting --------------")

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
airports_distances = nodes_distances(nodes_coords=airports_coords)

_logger.info("-------------- Define feasible paths --------------")
max_ground_distance = settings.ground_access_config.avg_speed * settings.ground_access_config.max_time / 60
# Identify which population cells are located near each airport, based on the maximum allowable ground distance.
population_cells_near_airports = get_pop_cells_near_airports(airports_coords=airports_coords,
                                                             population_coords=population_coords,
                                                             max_ground_distance=max_ground_distance)
destination_airports = []
for ii in population_cells_near_airports.keys():
    for destination_cell in settings.population_config.destination_cells:
        if destination_cell in population_cells_near_airports[ii]:
            destination_airports.append(ii)
_logger.info("For destination cells {}, the selected destination airports are {} based on the maximum ground distance."
             .format(settings.population_config.destination_cells, destination_airports))
# Create two graph to identify the edges above and below the distance threshold tau (single charge range)
airports_graph_below_tau = create_threshold_graph(distances=airports_distances,
                                                  tau=settings.aircraft_config.tau)
airports_graph_above_tau = create_threshold_graph(distances=airports_distances,
                                                  tau=settings.aircraft_config.tau, mode='above')

all_simple_paths = get_all_simple_path_to_destinations(graph=airports_graph_below_tau,
                                                       destination_nodes=destination_airports,
                                                       max_path_edges=settings.paths_config.max_edges)
attractive_simple_paths = get_attractive_paths(paths=all_simple_paths,
                                               distances=airports_distances,
                                               routing_factor_thr=settings.paths_config.routing_factor_thr)

pop_paths = get_pop_paths(pop_coords=population_coords, all_simple_paths=all_simple_paths,
                                                pop_cells_near_airports=population_cells_near_airports)


airports_df = pd.DataFrame({
    'id': range(settings.airports_config.num), 'type': 'airport', 'x': airports_coords[:, 0], 'y': airports_coords[:, 1],
    'population': 0
})

num_populations_cells = settings.population_config.cells_y * settings.population_config.cells_x

population_df = pd.DataFrame({
    'id': range(settings.airports_config.num, settings.airports_config.num+ num_populations_cells),
    'type': 'population', 'x': population_coords[:, 0], 'y': population_coords[:, 1],
    'population': get_pop_density(population_coords)
})

m = solve_eacn_model(airports_df, population_df, airports_graph_below_tau, all_simple_paths, pop_paths,
                     settings.aircraft_config.tau)
active_bases = []
if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:

    all_vars = m.getVars()
    y_vars = [v for v in all_vars if v.VarName.startswith('y[')]
    psi_vars = [v for v in all_vars if v.VarName.startswith('psi[')]

    active_bases = [int(v.VarName[2:-1]) for v in y_vars if v.X > 0.5]
    print(f" Basi di Ricarica Attive ({len(active_bases)}):")
    print(active_bases)

    print("\n-------------------------------------------")
    print(f"Valore Funzione Obiettivo: {m.ObjVal:,.2f}")
    print(f"MIP Gap: {m.MIPGap:.4%}")
    print("-------------------------------------------")

else:
    print("\nNessuna soluzione trovata. Stato Gurobi:", m.Status)

if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:

    all_vars = m.getVars()
    psi_vars = [v for v in all_vars if v.VarName.startswith('psi[')]

    active_path_indices = [int(v.VarName[4:-1]) for v in psi_vars if v.X > 0.5]
    solution_paths = [all_simple_paths[i] for i in active_path_indices]

    if not solution_paths:
        print("Nessun percorso è risultato fattibile nella soluzione trovata.")
else:
    print("Nessuna soluzione valida trovata. Impossibile visualizzare i percorsi.")

_logger.info("-------------- Plot --------------")
plot_dataset(population_coords=population_coords, population_density=population_density,
             airports_coords=airports_coords, airport_distances=airports_distances,
             graph_below_tau=airports_graph_below_tau, graph_above_tau=airports_graph_above_tau,
             destination_airports=destination_airports, destination_cells=settings.population_config.destination_cells,
             max_ground_distance = max_ground_distance, all_simple_paths=all_simple_paths,
             attractive_simple_paths = attractive_simple_paths,
             population_cells_near_airports = population_cells_near_airports, active_bases=active_bases)

