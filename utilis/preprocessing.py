import networkx as nx

def create_threshold_graph(dist: dict, tau: float, mode: str = "below") -> nx.Graph:
    """
    Creates a single undirected graph by including edges based on a distance threshold.

    Args:
        dist (dict): A dictionary where keys are (i, j) node pairs and values are the distances between them.
        tau (float): Distance threshold for including edges.
        mode (str): Determines which edges to include:
            - "below": includes edges with distance <= tau.
            - "above": includes edges with distance > tau.

    Returns:
        nx.Graph: A NetworkX graph containing the filtered edges.

    Raises:
        ValueError: If mode is not 'below' or 'above'.
    """
    if mode not in {"below", "above"}:
        raise ValueError("Mode must be 'below' or 'above'.")

    graph = nx.Graph()

    for (i, j), d in dist.items():
        if (mode == "below" and d <= tau) or (mode == "above" and d > tau):
            graph.add_edge(i, j, weight=d)

    return graph
