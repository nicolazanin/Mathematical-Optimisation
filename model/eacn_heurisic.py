
import random


def eacn_ks (airports_df, population_df, airports_graph_below_tau, all_simple_paths, pop_paths, tau, 
                            kernel_size=5, bucket_size=10, max_iterations=3):

    # Init
    kernel_airports = set(build_initial_kernel(airports_df, pop_paths, kernel_size))

    best_solution_vars = None
    best_obj_val = -float('inf') 

    airports = set(airports_df['id'].tolist())

    for iter in range(max_iterations): # manca maximum run time

        non_kernel_airports = airports.difference(kernel_airports)  
        non_kernel_airports_list = list(non_kernel_airports)
        
        random.shuffle(non_kernel_airports_list)
        # Creare insieme di bucket ecc ...


    return