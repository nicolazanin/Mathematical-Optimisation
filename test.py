import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB


from utilis.init_dataset import (cells_generation, nodes_generation, get_pop_cells_near_airports, get_pop_density,
                                 nodes_distances, grid_dimensions)
from utilis.preprocessing import create_threshold_graph
from utilis.init_model import get_all_simple_path_to_destinations, get_pop_paths
from utilis.plot import plot_dataset, plot_possible_paths
from utilis.eanc_reg_model import solve_eacn_model

# To achieve maximum scalability, the grid is built using the number of cells along the X and Y axes,
#  as well as the area of a single cell.
num_pop_cells_x = 15
num_pop_cells_y = 15
num_populations_cells = num_pop_cells_y * num_pop_cells_x
pop_cell_area = 2000

num_airports = 8
tau = 400

max_path_len = 3  # maximum of 3 adjacent edges (maximum of 2 intermediate airports)
cruise_speed = 400
ground_access_max_time = 90
ground_avg_speed = 60
max_ground_distance = ground_avg_speed * ground_access_max_time / 60

destination_cell = 61 + 15 # 61+15 to get multi airport as destination

pop_coords = cells_generation(num_pop_cells_x, num_pop_cells_y, pop_cell_area)
pop_density = get_pop_density(pop_coords)

total_width_pop_area, total_height_pop_area = grid_dimensions(num_pop_cells_x, num_pop_cells_y, pop_cell_area)
airports_coords = nodes_generation(num_airports, total_width_pop_area, total_height_pop_area)
airports_distances = nodes_distances(airports_coords)
pop_cells_near_airports = get_pop_cells_near_airports(airports_coords, pop_coords, max_ground_distance)

destination_airports = [ii for ii in pop_cells_near_airports.keys() if destination_cell in pop_cells_near_airports[ii]]
airports_graph_below_tau = create_threshold_graph(airports_distances, tau)
airports_graph_above_tau = create_threshold_graph(airports_distances, tau, mode='above')

plot_dataset(pop_coords=pop_coords, pop_density=pop_density, airports_coords=airports_coords,
             airport_distances=airports_distances, graph_below_tau=airports_graph_below_tau,
             graph_above_tau=airports_graph_above_tau, destination_airports=destination_airports,
             destination_cell=destination_cell, show_connections=True, show_density=True, show_colorbar=True)

all_simple_paths = get_all_simple_path_to_destinations(airports_graph_below_tau,destination_airports,max_path_len)
pop_paths = get_pop_paths(pop_coords=pop_coords, all_simple_paths=all_simple_paths,
                                                pop_cells_near_airports=pop_cells_near_airports)

for ii in range(len(all_simple_paths)):
    plot_possible_paths(pop_coords=pop_coords,airports_coords=airports_coords, destination_airports=destination_airports,
                       pop_paths=pop_paths,
                        graph=airports_graph_below_tau,paths=[all_simple_paths[ii]],
                       destination_cell=destination_cell, show_airports=True)

# plot_possible_paths(pop_coords=pop_coords,airports_coords=airports_coords, destination_airports=destination_airports,
#                    pop_paths=pop_paths,
#                     graph=airports_graph_below_tau,paths=all_simple_paths,
#                    destination_cell=destination_cell, show_airports=True)

# plt.show()

airports_df = pd.DataFrame({
    'id': range(num_airports), 'type': 'airport', 'x': airports_coords[:, 0], 'y': airports_coords[:, 1], 'population': 0
})

population_df = pd.DataFrame({
    'id': range(num_airports, num_airports + num_populations_cells),
    'type': 'population', 'x': pop_coords[:, 0], 'y': pop_coords[:, 1],
    'population': get_pop_density(pop_coords)
})


m = solve_eacn_model(airports_df, population_df, airports_graph_below_tau, all_simple_paths, pop_paths, tau)

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
    
        for i, single_path in enumerate(solution_paths):

            plot_possible_paths(
                pop_coords=pop_coords,
                airports_coords=airports_coords,
                destination_airports=destination_airports,
                graph=airports_graph_below_tau,
                paths=[single_path],
                destination_cell=destination_cell,
                pop_paths=pop_paths
            )
            plt.title(f'Visualizzazione del Percorso Scelto dalla Soluzione: {single_path}')
            
else:
    print("Nessuna soluzione valida trovata. Impossibile visualizzare i percorsi.")

plt.show()

print("End")
