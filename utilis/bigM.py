
def calculate_tight_big_m(airports_df, dist, G, tau):
    """
    Calcola i valori "stretti" per i parametri Big-M come descritto
    nella Sezione 3.1.1 del paper.
    """
    print("\nCalcolo dei parametri Big-M stretti...")
    
    airports = airports_df['id'].tolist()
    
    # Calcolo di M1 (un valore per ogni aeroporto, come da Formula 17)
    M1_vals = {}
    for i in airports:
        neighbors = list(G.neighbors(i))
        if not neighbors: # Areoporto è isolato, non ha nessun vicino a distanza < tau
            M1_vals[i] = 0 # Impostare M1_vals[i] = 0 è un modo efficiente per "comunicare" questa informazione al modello. Quando Gurobi vedrà questo valore, il vincolo rho[i] <= 0 * (1 - y[i]) forzerà rho_i a essere 0
        else:
            min_dist_to_neighbor = min(dist[i, j] for j in neighbors)
            # Questa è l'implementazione della logica della Proposizione 2
            M1_vals[i] = tau - min_dist_to_neighbor + 0.001 # + un piccolo epsilon

    # Calcolo di M2 e M3 (un valore per ogni tratta)
    M2_vals = {}
    M3_vals = {}
    for i, j in G.edges():
        edge = tuple(sorted((i,j)))
        
        # Prepara i dati per le formule
        neighbors_i = list(G.neighbors(i)) # N_i
        neighbors_j = list(G.neighbors(j)) # N_j

        # Calcolo M2 (Formula 18)
        if not neighbors_j:
            min_dist_from_j = 0 # Se la lista neighbors_j è vuota, il codice esegue min_dist_from_j = 0 e non prova a calcolare la min_dist_from_j che darebbe errore!
        else:
            min_dist_from_j = min(dist[j, r] for r in neighbors_j)
        M2_vals[edge] = dist[i,j] + tau - min_dist_from_j + 0.001

        # Calcolo M3 (Formula 19)
        if not neighbors_i:
            min_dist_from_i = 0 # Stessa cosa di sopra
        else:
            min_dist_from_i = min(dist[i, r] for r in neighbors_i)
        M3_vals[edge] = dist[i,j] + tau - min_dist_from_i - min_dist_from_j + 0.002
            
    print("Calcolo Big-M completato.")
    return M1_vals, M2_vals, M3_vals