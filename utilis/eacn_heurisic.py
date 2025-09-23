from collections import defaultdict
import random

def build_initial_kernel(airports_df, pop_paths, kernel_size):

    airports = airports_df['id'].to_numpy()

    airport_served_pops = defaultdict(set)
    airport_scores = {i: 0 for i in airports}

    for pop_id, paths in pop_paths.items():

        if not paths:
            continue

        for path in paths:
            for airport_node in path:
                airport_served_pops[airport_node].add(pop_id)

    for airport_id, pop_served in airport_served_pops.items():

        score = len(pop_served)
        airport_scores[airport_id] = score

    sorted_airports = sorted(airport_scores.items(), key=lambda item: item[1], reverse=True)

    initial_kernel_with_scores = sorted_airports[:kernel_size]
    initial_kernel = [airport_id for airport_id, score in initial_kernel_with_scores]

    return initial_kernel


def eacn_ks (airports_df, population_df, airports_graph_below_tau, all_simple_paths, pop_paths, tau, 
                            kernel_size=5, bucket_size=10, max_iterations=3):

    # Init
    kernel_airports = set(build_initial_kernel(airports_df, pop_paths, kernel_size))

    best_solution_vars = None
    best_obj_val = -float('inf') 

    airports = set(airports_df['id'].tolist())

    for iter in range(max_iterations):

        non_kernel_airports = airports.difference(kernel_airports)  
        non_kernel_airports_list = list(non_kernel_airports)
        
        random.shuffle(non_kernel_airports_list)
         

    

    return