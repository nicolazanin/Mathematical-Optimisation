import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize


def plot_population_grid(ax, pop_coords, pop_density=None, destination=None,
                         show_colorbar=True, show_density=True):
    """
    Plot population cells as colored rectangles representing population density.
    """
    pop_coords = np.array(pop_coords)
    cell_size = abs(float(pop_coords[1][0] - pop_coords[0][0]))

    # Define color map from green to orange
    colors = [(0.8, 1, 0.8, 0.5), (1, 0.7, 0.4, 0.5)]
    cmap = LinearSegmentedColormap.from_list("green_orange", colors)

    if show_density and pop_density is not None:
        norm = Normalize(vmin=min(pop_density), vmax=max(pop_density))
    else:
        norm = None  # no normalization needed if no density color
    for idx, coord in enumerate(pop_coords):
        x = coord[0] - cell_size / 2
        y = coord[1] - cell_size / 2

        if show_density and pop_density is not None:
            density = pop_density[idx]
            color = cmap(norm(density))
        else:
            color = (0.8, 0.8, 0.8, 0.1)  # light grey, transparent for uniform cells

        hatch = '///' if destination is not None and idx == destination else None

        rect = plt.Rectangle((x, y), cell_size, cell_size, facecolor=color, hatch=hatch)
        ax.add_patch(rect)

    # Scatter cell centers (optional)
    ax.scatter(pop_coords[:, 0], pop_coords[:, 1], s=10, c='grey', marker='o', label='Population Cells')

    # Show colorbar only if density coloring is on
    if show_density and show_colorbar and pop_density is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Densità di Popolazione')


def plot_airports(ax, airports_coords, label_airports=True):
    """
    Plot airport locations with optional labels.
    """
    airports_coords = np.array(airports_coords)
    ax.scatter(airports_coords[:, 0], airports_coords[:, 1], s=40, c='gray', marker='x', label='Airports', zorder=5)

    if label_airports:
        for idx, coords in enumerate(airports_coords):
            ax.text(coords[0], coords[1] + 5, str(idx), fontsize=11, color='blue')


def plot_airports_destination(ax, airports_coords, destination_airport=None):
    """
    Plot airport locations. Highlight airports in the 'destination' list.

    Parameters:
    - ax: matplotlib axis
    - airports_coords: array-like, shape (n_airports, 2)
    - destination: list of airport indices to highlight (can be empty or None)
    """
    airports_coords = np.array(airports_coords)

    if destination_airport:
        dest_coords = airports_coords[destination_airport]
        ax.scatter(dest_coords[:, 0], dest_coords[:, 1], s=40,
                   facecolors='none', edgecolors='purple', linewidths=1,
                   marker='o', label='Destination Airports', zorder=5)


def plot_connections(ax, airports_coords, airport_distances, graph_below_tau=None, graph_above_tau=None):
    """
    Plot airport connections below and above tau.
    """
    pos = {idx: (coords[0], coords[1]) for idx, coords in enumerate(airports_coords)}

    # Edges and labels
    if graph_below_tau:
        edge_labels = {}
        for u, v in graph_below_tau.edges():
            try:
                edge_labels[(u, v)] = f"{airport_distances[u, v]:.0f} km"
            except KeyError:
                continue
        nx.draw_networkx_edges(graph_below_tau, pos, edge_color='gray', style='dotted', ax=ax)
        nx.draw_networkx_edge_labels(graph_below_tau, pos, edge_labels=edge_labels, font_color='gray',
                                     font_size=7, ax=ax)

    if graph_above_tau:
        edge_labels = {}
        for u, v in graph_above_tau.edges():
            try:
                edge_labels[(u, v)] = f"{airport_distances[u, v]:.0f} km"
            except KeyError:
                continue
        nx.draw_networkx_edges(graph_above_tau, pos, edge_color='red', style='dotted', ax=ax)
        nx.draw_networkx_edge_labels(graph_above_tau, pos, edge_labels=edge_labels, font_color='red',
                                     font_size=7, ax=ax)


def plot_paths(ax, airports_coords, graph: nx.Graph, paths: list, color='blue'):
    """
    Plots paths on a given matplotlib axis.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        graph (nx.Graph): The NetworkX graph to provide full context (for background edges).
        paths (list): A list of node IDs representing the path.
        color (str): Color of the path line.
    """
    pos = {idx: (coords[0], coords[1]) for idx, coords in enumerate(airports_coords)}
    # Plot full graph edges in light gray dotted style for context
    nx.draw_networkx_edges(graph, pos=pos, edge_color='gray', style='dotted', ax=ax)

    colors = plt.cm.get_cmap('tab20', len(paths))

    for i, path in enumerate(paths):
        # Edges for this path
        path_edges = list(zip(paths[i][:-1], paths[i][1:]))

        # Draw edges of this path with unique color
        nx.draw_networkx_edges(
            graph, pos=pos, edgelist=path_edges, edge_color=[colors(i)], width=2.5, ax=ax
        )

def show_pop_paths(ax, pop_coords, paths: list, pop_paths:dict, color='purple'):
    """
    Shows population cells that are related to the paths.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        paths (list): A list of node IDs representing the path.
        color (str): Color of the path line.
    """
    legend_activate = False
    for path in paths:
        for idx in pop_paths.keys():
            if path in pop_paths[idx]:
                ax.scatter(pop_coords[idx, 0], pop_coords[idx, 1], s=30, c=color, marker='o',
                           label="Population of the path" if not legend_activate else None, zorder=7)
                legend_activate = True


def plot_dataset(pop_coords=None, pop_density=None, airports_coords=None,
                 airport_distances=None, graph_below_tau=None, graph_above_tau=None, destination_airports=None,
                 destination_cell=None, show_grid=True, show_airports=True, show_connections=True,
                 show_colorbar=True, show_density=True):
    """
    Main function to plot the dataset with optional layers.
    """
    fig, ax = plt.subplots(figsize=(15, 10))

    if show_grid and pop_coords is not None:
        plot_population_grid(ax, pop_coords, pop_density, destination=destination_cell, show_colorbar=show_colorbar,
                             show_density=show_density)

    if show_airports and airports_coords is not None:
        plot_airports(ax, airports_coords)

    if show_airports and destination_airports is not None:
        plot_airports_destination(ax, airports_coords, destination_airports)

    if show_connections:
        plot_connections(ax, airports_coords, airport_distances,
                         graph_below_tau=graph_below_tau,
                         graph_above_tau=graph_above_tau)

    ax.set_title('Mappa degli Aeroporti, Popolazione e Connessioni Possibili')
    ax.set_xlabel('Coordinata X (km)')
    ax.set_ylabel('Coordinata Y (km)')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')


def plot_possible_paths(pop_coords=None, airports_coords=None, destination_airports=None,
                       pop_paths=None,
                       graph=None, paths=None,
                       destination_cell=None, show_airports=True):
    """
    Main function to plot the possible paths.
    """
    fig, ax = plt.subplots(figsize=(15, 10))

    if pop_coords is not None:
        plot_population_grid(ax=ax, pop_coords=pop_coords, pop_density=None, destination=destination_cell)

    if show_airports and airports_coords is not None:
        plot_airports(ax, airports_coords)

    if show_airports and destination_airports is not None:
        plot_airports_destination(ax, airports_coords, destination_airports)

    if paths is not None and graph is not None:
        plot_paths(ax, airports_coords, graph, paths, pop_paths)

    if pop_paths is not None:
        show_pop_paths(ax,pop_coords,paths,pop_paths,color='purple')

    ax.set_title('Mappa degli Aeroporti e Connessioni Possibili')
    ax.set_xlabel('Coordinata X (km)')
    ax.set_ylabel('Coordinata Y (km)')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
