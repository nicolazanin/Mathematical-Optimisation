import numpy as np
from gurobipy import GRB
from datetime import datetime
import logging
import time

from utils.init_dataset import (cells_generation, nodes_generation, get_population_cells_near_airports,
                                get_pop_density, get_nodes_distances_alt, get_grid_dimensions,
                                get_activation_cost_airports, get_destinations_airports_info,
                                get_population_cells2airports_distances)
from utils.preprocessing import (get_threshold_graph, get_attractive_paths_from_rft, get_all_paths_to_destinations,
                                 get_population_cells_paths, get_population_cells_too_close_to_destination_cells,
                                 get_airports_too_close_to_destination_cells, get_attractive_paths,
                                 get_attractive_graph)
from model.utils_model import get_outputs_from_model
from model.eanc_reg_model import solve_eacn_model
from utils.settings import setup_logging, settings
from utils.scalability_utils import apply_preset, init_results_dict, print_report

settings = settings.from_yaml("config_scalability.yml")


def scalability(tau: int, num: int, cell_x: int, cell_y: int, cell_area: int, routing_factor_thr: float) -> None:
    tic = time.time()

    setup_logging(log_prefix="EACN_REG", print_file=settings.print_logs)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _logger = logging.getLogger(__name__)
    _logger.setLevel(settings.logging_lvl)

    settings.airports_config.num = num
    settings.aircraft_config.tau = tau
    settings.population_config.cells_x = cell_x
    settings.population_config.cells_y = cell_y
    settings.population_config.cell_area = cell_area
    settings.paths_config.routing_factor_thr = routing_factor_thr

    _logger.info("-------------- EACN-REG scalability starting --------------")
    _logger.info("-------------- N: {} --------------".format(settings.airports_config.num))
    _logger.info("-------------- K: {} --------------".format(settings.population_config.cells_x *
                                                              settings.population_config.cells_y))
    _logger.info("-------------- tau: {} --------------".format(settings.aircraft_config.tau))
    _logger.info("-------------- routing_factor: {} --------------".format(settings.paths_config.routing_factor_thr))

    results = init_results_dict()
    for i in range(3):
        results["test"].append(i)
        results["K"].append(settings.population_config.cells_x * settings.population_config.cells_y)
        results["N"].append(settings.airports_config.num)
        results["tau"].append(settings.aircraft_config.tau)
        settings.random_seed = i
        np.random.seed(settings.random_seed)
        settings.population_config.destination_cells = [int(
            str(np.random.randint(settings.population_config.cells_y * 0.2, settings.population_config.cells_y * 0.8)) +
            str(np.random.randint(settings.population_config.cells_x * 0.2, settings.population_config.cells_x * 0.8)))]

        _logger.info("-------------- Initialize the population grid dataset --------------")
        population_coords = cells_generation(num_cells_x=settings.population_config.cells_x,
                                             num_cells_y=settings.population_config.cells_y,
                                             cell_area=settings.population_config.cell_area)
        population_density = get_pop_density(population_coords=population_coords,
                                             min_density=settings.population_config.min_density,
                                             max_density=settings.population_config.max_density,
                                             high_population_cells=settings.population_config.high_population_cells)

        _logger.info("-------------- Initialize the airports dataset --------------")
        total_width_pop_area, total_height_pop_area = get_grid_dimensions(
            num_cells_x=settings.population_config.cells_x,
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
        airports_distances_alt = get_nodes_distances_alt(nodes_coords=airports_coords, res=settings.paths_config.res)

        _logger.info("-------------- Define Destination Airport/s --------------")
        max_ground_distance = settings.ground_access_config.avg_speed * settings.ground_access_config.max_time / 60
        population_cells_near_airports = get_population_cells_near_airports(airports_coords=airports_coords,
                                                                            population_coords=population_coords,
                                                                            max_ground_distance=max_ground_distance)

        population_cells2airport_distances = get_population_cells2airports_distances(
            population_coords=population_coords,
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
        population_cells_too_close_to_destination_cells = (
            get_population_cells_too_close_to_destination_cells(population_coords=population_coords,
                                                                destination_cells=settings.population_config.destination_cells,
                                                                min_distance_to_destination_cells=min_distance_to_destination_cells))
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
        for test_name in ["b&c", "kn_1", "kn_3"]:
            apply_preset(settings, "scalability_tests.yml", test_name)

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
            population_cells_covered_close_dest = [int(cell) for cells in
                                                   population_cells_too_close_to_destination_cells.values()
                                                   for cell in cells]
            charging_airports = []
            population_covered = []
            bound = 0
            if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
                charging_airports, population_from_dest, active_path_indices, bound = get_outputs_from_model(m)
                population_covered = population_cells_covered_close_dest + population_from_dest
                _logger.info("Charging airports: {} ({})".format(str(charging_airports), len(charging_airports)))
                _logger.info("Population covered: {} ({})".format(str(population_covered), len(population_covered)))
            else:
                _logger.info("No solution was found. Status:".format(m.Status))

            results[test_name]["obj_1"].append(len(population_covered))
            results[test_name]["obj_2"].append(len(charging_airports))
            results[test_name]["bound"].append(round(bound, 2))
            results[test_name]["t"].append(round(time_exec, 2))
            if test_name == "b&c":
                cols = m.NumVars
                rows = m.NumConstrs
                results[test_name]["cols"].append(cols)
                results[test_name]["rows"].append(rows)

    _logger.info("Total execution time for EACN-REG: {:.1f} minutes".format((time.time() - tic) / 60))

    print_report(results, f"EACN_REG_{timestamp}")


if __name__ == "__main__":
    scalability(num=50, cell_x=10, cell_y=10, tau=400, cell_area=4500, routing_factor_thr=1.4)
    scalability(num=50, cell_x=10, cell_y=10, tau=600, cell_area=4500, routing_factor_thr=1.2)
    scalability(num=50, cell_x=10, cell_y=10, tau=800, cell_area=4500, routing_factor_thr=1.2)
    scalability(num=50, cell_x=10, cell_y=20, tau=400, cell_area=2250, routing_factor_thr=1.4)
    scalability(num=50, cell_x=10, cell_y=20, tau=600, cell_area=2250, routing_factor_thr=1.2)
    scalability(num=50, cell_x=10, cell_y=20, tau=800, cell_area=2250, routing_factor_thr=1.2)
    scalability(num=100, cell_x=10, cell_y=10, tau=400, cell_area=4500, routing_factor_thr=1.4)
    scalability(num=100, cell_x=10, cell_y=10, tau=600, cell_area=4500, routing_factor_thr=1.2)
    scalability(num=100, cell_x=10, cell_y=10, tau=800, cell_area=4500, routing_factor_thr=1.2)
    scalability(num=100, cell_x=10, cell_y=20, tau=400, cell_area=2250, routing_factor_thr=1.4)
    scalability(num=100, cell_x=10, cell_y=20, tau=600, cell_area=2250, routing_factor_thr=1.2)
    scalability(num=100, cell_x=10, cell_y=20, tau=800, cell_area=2250, routing_factor_thr=1.2)
