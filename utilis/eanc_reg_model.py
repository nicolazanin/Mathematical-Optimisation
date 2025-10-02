import numpy as np
import gurobipy as gp
from gurobipy import GRB

def calculate_tight_big_m(airports_df, dist, G, tau):
    
    airports = airports_df['id'].tolist()
    
    M1_vals = {}
    for i in airports:
        neighbors = list(G.neighbors(i))
        if not neighbors: 
            M1_vals[i] = 0 
        else:
            min_dist_to_neighbor = min(dist[i, j] for j in neighbors)

            M1_vals[i] = tau - min_dist_to_neighbor + 0.001 

    M2_vals = {}
    M3_vals = {}
    for i, j in G.edges():
        edge = tuple(sorted((i,j)))
        
        neighbors_i = list(G.neighbors(i))
        neighbors_j = list(G.neighbors(j))

        if not neighbors_j:
            min_dist_from_j = 0 
        else:
            min_dist_from_j = min(dist[j, r] for r in neighbors_j)
        M2_vals[edge] = dist[i,j] + tau - min_dist_from_j + 0.001

        if not neighbors_i:
            min_dist_from_i = 0 
        else:
            min_dist_from_i = min(dist[i, r] for r in neighbors_i)
        M3_vals[edge] = dist[i,j] + tau - min_dist_from_i - min_dist_from_j + 0.002
            
    return M1_vals, M2_vals, M3_vals


def solve_eacn_model(airports_df, population_df, airports_graph_below_tau, all_simple_paths, pop_paths, tau, 
                     ks=False, kernel=None, bucket=None, best_obj_value=-float('inf')):

    if ks:
        candidate_airports = kernel.union(bucket) # MODIFICA KS

    mu1 = 100
    mu2 = 10000

    airports = airports_df['id'].tolist() 
    coords = airports_df[['x', 'y']].to_numpy() 
    dist = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=2)

    M1_vals, M2_vals, M3_vals = calculate_tight_big_m(airports_df, dist, airports_graph_below_tau, tau)

    m = gp.Model("EACN_REG_Strengthened")
    m.setParam('LogToConsole', 0)
    y = m.addVars(airports, vtype=GRB.BINARY, name="y")
    rho = m.addVars(airports, vtype=GRB.CONTINUOUS, name="rho", lb=0.0)
    w = m.addVars([(i, j) for i in airports for j in airports_graph_below_tau.neighbors(i)], vtype=GRB.BINARY, name="w") # -> Correzione chagpt
    chi = m.addVars(airports, vtype=GRB.BINARY, name="chi")

    canonical_edges = sorted(list(set(tuple(sorted(edge)) for edge in airports_graph_below_tau.edges())))

    z = m.addVars(canonical_edges, vtype=GRB.BINARY, name="z")

    psi = m.addVars(len(all_simple_paths), vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="psi")
    phi = m.addVars(population_df['id'], vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="phi")

    population_covered = gp.quicksum(phi[k] * population_df.set_index('id').loc[k, 'population'] for k in population_df['id'])
    installation_cost = gp.quicksum(y[i] for i in airports)
    objective_func = mu1 * population_covered - mu2 * installation_cost 
    m.setObjective(objective_func, GRB.MAXIMIZE) 

    if ks:
        m.addConstr(objective_func >= best_obj_value + 0.001, name="obj_improvement_constraint") # MODIFICA KS

    for i in airports:
        if ks:
            if i not in candidate_airports: # MODIFICA KS
                y[i].ub = 0
        neighbors = list(airports_graph_below_tau.neighbors(i))
        m.addConstr(rho[i] <= M1_vals[i] * (1 - y[i])) 
        m.addConstr(y[i] + gp.quicksum(w.get((i,j), 0) for j in neighbors) + chi[i] == 1) 
        m.addConstr(rho[i] >= M1_vals[i] * chi[i])
        for j in neighbors:
            edge = tuple(sorted((i,j)))
            m.addGenConstrIndicator(w.get((i,j),0), 1, rho[i] <= dist[i, j] + rho[j]) 
            m.addConstr(rho[i] >= dist[i, j] + rho[j] - M2_vals[edge] * (1 - w.get((i,j), 0)))

    for i, j in canonical_edges:
        m.addConstr(dist[i, j] + rho[i] + rho[j] <= tau + M3_vals[i,j] * (1 - z[i,j])) 

    for i in airports:
        neighbors = list(airports_graph_below_tau.neighbors(i))
        if neighbors:
            min_dist_to_neighbor = min(dist[i, j] for j in neighbors)
            m.addConstr(rho[i] >= min(min_dist_to_neighbor, M1_vals[i]) * (1 - y[i])) 

    for i, j in canonical_edges:
        m.addConstr(z[i,j] <= 1 - chi[i]) 
        m.addConstr(z[i,j] <= 1 - chi[j]) 

    for p_idx, path in enumerate(all_simple_paths):
        path_edges = [tuple(sorted((path[i], path[i+1]))) for i in range(len(path) - 1)]
        m.addConstr(psi[p_idx] - gp.quicksum(z.get(edge, 0) for edge in path_edges) >= 1 - len(path_edges)) 

    for i, j in canonical_edges:
        M4_ij = tau - dist[i,j]
        m.addConstr(dist[i, j] + rho[i] + rho[j] + M4_ij * z[i,j] >= tau)

    for p_idx, path in enumerate(all_simple_paths):
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge = tuple(sorted((u,v)))
            m.addConstr(psi[p_idx] <= z[edge])

    for pop_id in population_df['id']:
        paths_for_pop = pop_paths.get(pop_id, [])
        path_indices = [i for i, p in enumerate(all_simple_paths) if p in paths_for_pop]
        if path_indices:
            m.addConstr(phi[pop_id] <= gp.quicksum(psi[p_idx] for p_idx in path_indices))
        else:
            m.addConstr(phi[pop_id] == 0)

    m.setParam('TimeLimit', 600)
    m.optimize()

    return m


