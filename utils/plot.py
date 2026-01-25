import plotly.graph_objects as go
import numpy as np
import networkx as nx
import plotly.io as pio
import datetime

pio.renderers.default = "browser"


def get_population_heatmap(population_coords: np.ndarray, population_density: np.ndarray) -> go.Heatmap:
    """
    Returns a Heatmap object that shows the population density of each cell.

    Args:
        population_coords (np.ndarray): A NumPy array of shape (num_population_cells, 2) containing (x, y) coordinates
            of population cell centers.
        population_density (np.ndarray): A NumPy array of population density values (integers) with the same length as
            `population_coords`.

    Returns:
        go.Heatmap: A Heatmap object that represents the population density.
    """

    x_unique = np.unique(population_coords[:, 0])
    y_unique = np.unique(population_coords[:, 1])

    grid_width = len(x_unique)
    grid_height = len(y_unique)

    # Reshape into 2D
    z_grid = population_density.reshape((grid_height, grid_width))
    custom_colorscale = [
        [0.0, 'rgba(220,30,0,0)'],
        [0.25, 'rgba(225,30,0,0.15)'],
        [0.5, 'rgba(230,30,0,0.2)'],
        [0.75, 'rgba(235,30,0,0.3)'],
        [1.0, 'rgba(240,30,0,0.4)']
    ]

    density_heatmap = go.Heatmap(
        z=z_grid,
        x=np.sort(x_unique),
        y=np.sort(y_unique),
        colorscale=custom_colorscale,
        colorbar=dict(title=dict(
            text="Population Density [people/km²]: ",
            font=dict(
                size=12,
            ),
        ), orientation='h',
            x=0.5,
            y=-0.03,
            xanchor='center',
            yanchor='top',
            thickness=10,
            len=0.5
        ),
        zmin=population_density.min(),
        zmax=population_density.max(),
        name="Population Cell",
        hovertemplate="Population density: %{z} people/km² <extra></extra>",
        legend="legend1",
        visible=True,
    )

    return density_heatmap


def get_population_cells(population_coords: np.ndarray) -> go.Scatter:
    """
    Returns a Scatter object that shows the population cells.

    Args:
        population_coords (np.ndarray): A NumPy array of shape (num_population_cells, 2) containing (x, y) coordinates
            of population cell centers.

    Returns:
        go.Scatter: A Scatter object that shows the population cells.
    """
    labels = [str(num) for num in range(len(population_coords))]
    population_cell_markers = go.Scatter(
        x=population_coords[:, 0],
        y=population_coords[:, 1],
        mode='markers',
        marker=dict(size=5, color='rgba(128, 128, 128, 0.5)'),
        name='Population Cells',
        customdata=labels,
        showlegend=True,
        hovertemplate="Population Cell: %{customdata}<br>"
                      "Coordinates: %{x:.1f}km x %{y:.1f}km <extra></extra>",
        legend="legend1",
        legendrank=1
    )

    return population_cell_markers


def get_destination_cells(population_coords: np.ndarray, destination_cells: list) -> go.Scatter:
    """
    Returns a Scatter object that shows the destination cells.

    Args:
        population_coords (np.ndarray): A NumPy array of shape (num_population_cells, 2) containing (x, y) coordinates
            of population cell centers.
        destination_cells (list): An array of destination cell indices.

    Returns:
        go.Scatter: A Scatter object that shows the destination cells.
    """
    labels = [str(num) for num in range(len(destination_cells))]
    destination_cell_markers = go.Scatter(
        x=population_coords[destination_cells, 0], y=population_coords[destination_cells, 1],
        mode='markers',
        marker=dict(
            color='rgba(135, 206, 250, 0.5)',
            size=20,
            line=dict(
                color='rgba(147, 112, 219, 1)',
                width=2
            )
        ),
        name="Destination Cells",
        customdata=labels,
        hovertemplate="Destination Cell: %{customdata} <extra></extra>",
        showlegend=True,
        legend="legend1",
        legendrank=2
    )

    return destination_cell_markers


def get_max_ground_dist_destination_cells(population_coords: np.ndarray, destination_cells: list,
                                          max_ground_distance: float) -> list[go.Scatter]:
    """
    Returns a list of Scatter objects representing the circular area of 'max ground distance' from each destination
    cell.

    Args:
        population_coords (np.ndarray): A NumPy array of shape (num_population_cells, 2) containing (x, y) coordinates
            of population cell centers.
        destination_cells (list): An array of destination cell indices.
        max_ground_distance (float): Maximum allowed ground distance to consider a population cell "near" an airport.

    Returns:
        list: A list of Scatter objects representing the circular area of 'max ground distance' from each destination
        cell.
    """
    circle_traces = []

    for i, idx in enumerate(destination_cells):
        x_center, y_center = population_coords[idx]
        theta = np.linspace(0, 2 * np.pi, 200)
        x_circle = x_center + max_ground_distance * np.cos(theta)
        y_circle = y_center + max_ground_distance * np.sin(theta)
        circle_trace = go.Scatter(
            x=x_circle,
            y=y_circle,
            mode='lines',
            line=dict(color='rgba(147, 112, 219, 0.5)', width=2, dash='dot'),
            name="Max Ground Distance From Destination Cells" if i == 0 else None,
            hovertemplate="Population Cell: {}<br>Max Ground Distance: {}km<extra></extra>".format(idx,
                                                                                                   max_ground_distance),
            showlegend=(i == 0),
            legend="legend1",
            legendgroup="max_ground_distance",
            legendrank=3,
            visible='legendonly',
        )
        circle_traces.append(circle_trace)

    return circle_traces


def get_min_dist_destination_cells(population_coords: np.ndarray, destination_cells: list,
                                   min_distance_to_destination_cells: float) -> list[go.Scatter]:
    """
    Returns a list of Scatter objects representing the circular area of 'min_distance_to_destination_cells'
    from each destination cell.

    Args:
        population_coords (np.ndarray): A NumPy array of shape (num_population_cells, 2) containing (x, y) coordinates
            of population cell centers.
        destination_cells (list): An array of destination cell indices.
        min_distance_to_destination_cells(float): Minimum allowed distance to consider a population cell or an airport
            from the destination cells (km).
    Returns:
        list: A list of Scatter objects representing the circular area of 'min_distance_to_destination_cells' from each
            destination cell.
    """
    circle_traces = []

    for i, idx in enumerate(destination_cells):
        x_center, y_center = population_coords[idx]
        theta = np.linspace(0, 2 * np.pi, 200)
        x_circle = x_center + min_distance_to_destination_cells * np.cos(theta)
        y_circle = y_center + min_distance_to_destination_cells * np.sin(theta)
        circle_trace = go.Scatter(
            x=x_circle,
            y=y_circle,
            mode='lines',
            fill='toself',
            fillcolor='rgba(147, 112, 219, 0.15)',
            line=dict(color='rgba(147, 112, 219, 0.5)', width=2, dash='dot'),
            name="Min Distance From Destination Cells" if i == 0 else None,
            hovertemplate="Population Cell: {}<br>Min Distance: {}km<extra></extra>".format(idx,
                                                                                    min_distance_to_destination_cells),
            showlegend=(i == 0),
            legend="legend1",
            legendgroup="min_distance",
            legendrank=4,
            visible='legendonly',
        )
        circle_traces.append(circle_trace)

    return circle_traces


def get_population_grid(population_coords: np.ndarray) -> list[dict]:
    """
    Returns a list of dictionaries (shapes) representing the rectangular shapes of the population grid.

    Args:
        population_coords (np.ndarray): A NumPy array of shape (num_population_cells, 2) containing (x, y) coordinates
            of population cell centers.

    Returns:
        list: A list of dictionaries (shapes) representing the rectangular shapes of the population grid.
    """
    population_grid = []
    cell_size = abs(float(population_coords[1][0] - population_coords[0][0]))

    for coord in population_coords:
        x, y = coord[0], coord[1]
        x0 = x - cell_size / 2
        y0 = y - cell_size / 2
        x1 = x + cell_size / 2
        y1 = y + cell_size / 2

        shape = {
            "type": "rect",
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "line": {"color": "rgba(89, 89, 89, 0.8)", "width": 0.5},
        }
        population_grid.append(shape)

    return population_grid


def get_airports(airports_coords: np.ndarray) -> go.Scatter:
    """
    Returns a Scatter object that shows the airport nodes.

    Args:
        airports_coords (np.ndarray): A NumPy array of shape (num_airports, 2) containing (x, y) coordinates of each
            airport.

    Returns:
        go.Scatter: A Scatter that shows the airport nodes.
    """
    labels = [str(num) for num in range(len(airports_coords))]
    airport_markers = go.Scatter(
        x=airports_coords[:, 0],
        y=airports_coords[:, 1],
        mode='markers',
        marker=dict(
            color='rgba(250, 154, 255,0.3)',
            size=10,
            line=dict(
                color='rgb(250, 97, 255)',
                width=3
            )
        ),
        name="Airports " + "✈️",
        legendgroup="airports",
        customdata=labels,
        hovertemplate="Airport: %{customdata}<extra></extra>",
        showlegend=True,
        legend="legend1",
        legendrank=5
    )

    return airport_markers


def get_airport_icons(airports_coords: np.ndarray) -> go.Scatter:
    """
    Returns a Scatter object that shows the airport nodes through emoji.

    Args:
        airports_coords (np.ndarray): A NumPy array of shape (num_airports, 2) containing (x, y) coordinates of each
            airport.

    Returns:
        go.Scatter: A Scatter that shows the airport nodes through emoji.
    """
    emoji = "✈️"
    airport_icons = go.Scatter(
        x=airports_coords[:, 0] + 20,
        y=airports_coords[:, 1],
        mode='text',
        text=[emoji] * len(airports_coords[:, 0]),
        textfont=dict(
            size=20,
        ),
        name="Airports",
        legendgroup="airports",
        showlegend=False,
        hoverinfo="none",
        legend="legend1"
    )

    return airport_icons


def get_charging_airports(airports_coords: np.ndarray, charging_airports: list) -> go.Scatter:
    """
    Returns a Scatter object that shows the charging airport nodes.

    Args:
        airports_coords (np.ndarray): A NumPy array of shape (num_airports, 2) containing (x, y) coordinates of each
            airport.
        charging_airports (list): A list of airports indices representing the charging bases.

    Returns:
        go.Scatter: A Scatter that shows the airport nodes.
    """
    labels = [str(num) for num in charging_airports]
    charging_airport_markers = go.Scatter(
        x=airports_coords[charging_airports, 0],
        y=airports_coords[charging_airports, 1],
        mode='markers',
        marker=dict(
            color='rgb(141, 204, 71)',
            size=10,
            line=dict(
                color='rgb(141, 204, 71)',
                width=3
            )
        ),
        name="Charging Airports " + "🔋",
        legendgroup="charging_airports",
        showlegend=True,
        visible='legendonly',
        customdata=labels,
        hovertemplate="Charging Airport: %{customdata}<extra></extra>",
        legend="legend5",
    )
    return charging_airport_markers


def get_charging_airport_icons(airports_coords: np.ndarray, charging_airports: list) -> go.Scatter:
    """
    Returns a Scatter object that shows the charging airport nodes through emoji.

    Args:
        airports_coords (np.ndarray): A NumPy array of shape (num_airports, 2) containing (x, y) coordinates of each
            airport.
        charging_airports (list): A list of airports indices representing the bases.

    Returns:
        go.Scatter: A Scatter that shows the airport nodes through emoji.
    """
    emoji = "🔋"
    charging_airport_icons = go.Scatter(
        x=airports_coords[charging_airports, 0] - 20,
        y=airports_coords[charging_airports, 1],
        mode='text',
        text=[emoji] * len(airports_coords[charging_airports, 0]),
        textfont=dict(
            size=30,
        ),
        name="Charging Stations",
        showlegend=False,
        legend="legend5",
        legendgroup="charging_airports",
        visible='legendonly',
        hoverinfo="none",
    )

    return charging_airport_icons


def get_destination_airports(airports_coords: np.ndarray, destination_airports: np.ndarray) -> go.Scatter:
    """
    Returns a Scatter object that shows destination airport nodes.

    Args:
        airports_coords (np.ndarray): A NumPy array of shape (num_airports, 2) containing (x, y) coordinates of each
            airport.
        destination_airports (np.ndarray): A NumPy array of destination airports indices close to the destination cells.

    Returns:
        go.Scatter: A Scatter that shows destination airport nodes.
    """
    labels = np.array([[str(num) for num in range(len(destination_airports))],
                       [str(num) for num in destination_airports]]).T
    destination_airport_markers = go.Scatter(
        x=airports_coords[list(destination_airports), 0],
        y=airports_coords[list(destination_airports), 1],
        mode='markers',
        marker=dict(
            color='rgb(250, 97, 255)',
            size=10,
            line=dict(
                color='rgb(250, 97, 255)',
                width=3
            )
        ),
        name="Destination Airports " + "✈️",
        customdata=labels,
        hovertemplate="Airport: %{customdata[1]}<br>"
                      "Destination Airport: %{customdata[0]}<extra></extra>",
        showlegend=True,
        legend="legend1",
        legendrank=6
    )

    return destination_airport_markers


def get_connections(airports_coords: np.ndarray, airport_distances: dict, graph: nx.Graph, above: bool = True) -> list[
    go.Scatter]:
    """
    Returns a list of Scatter objects representing the connection between airports nodes. Different colors are used to
    show the feasible connection (below tau) and unfeasible connection (above tau).

    Args:
        airports_coords (np.ndarray): A NumPy array of shape (num_airports, 2) containing (x, y) coordinates of each
            airport.
        airport_distances (dict): A dictionary where each key is a tuple (i, j) representing a pair of node indices,and
            the value is the Euclidean distance between node i and node j.
        graph (nx.Graph): A NetworkX graph.
        above (bool): If the connections to represent are above or below tau.

    Returns:
        list: A list of Scatter objects representing the connection between airports nodes.
    """
    if not above:
        color = "rgba(99, 99, 99, 0.6)"
        name = "Connections Below Tau"
        visible = True
    else:
        color = "rgba(207, 13, 13, 0.5)"
        name = "Connections Above Tau"
        visible = 'legendonly'
    connections = []
    for i, (u, v) in enumerate(graph.edges()):
        x0, y0 = airports_coords[u]
        x1, y1 = airports_coords[v]
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        if u < v:
            text = "{:.0f}".format(airport_distances[u, v])
        else:
            text = "{:.0f}".format(airport_distances[v, u])
        connections.append(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(color=color, dash='dot', width=2),
            name=name if i == 0 else None,
            legendgroup=name,
            showlegend=(i == 0),
            legend="legend2",
            hoverinfo="none",
            visible=visible,
        ))
        connections.append(go.Scatter(
            x=[xm],
            y=[ym],
            mode='text',
            text=[text + "km"],
            textposition="middle left",
            textfont=dict(
                size=12,
                color=color,
                weight="bold"
            ),
            legendgroup=name,
            showlegend=False,
            legend="legend2",
            visible=visible,
            hovertemplate="Connection: {}<br>".format(str((u, v))) +
                          "Distance: {}<extra></extra>".format(text + "km"),
        ))

    return connections


def get_paths(airports_coords: np.ndarray, paths: np.ndarray, attractive: bool = True) -> list[go.Scatter]:
    """
    Returns a list of Scatter objects representing paths between airport nodes. Paths are categorized as either
    attractive paths or all paths, with distinct legends used to differentiate them.

    Args:
        airports_coords (np.ndarray): A NumPy array of shape (num_airports, 2) containing (x, y) coordinates of each
            airport.
        paths (np.ndarray): A NumPy array of paths (each path is a list of node IDs).
        attractive (bool): If the paths to represent are the attractive paths or all the paths.

    Returns:
        list: A list of Scatter objects representing the connection between airports nodes.
    """
    paths_lines = []
    if attractive:
        legend = "legend4"
    else:
        legend = "legend3"
    for i, path in enumerate(paths):
        edges = list(zip(path[:-1], path[1:]))
        x = []
        y = []
        for u, v in edges:
            x.append(airports_coords[u, 0])
            x.append(airports_coords[v, 0])
            y.append(airports_coords[u, 1])
            y.append(airports_coords[v, 1])
        paths_lines.append(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(width=3),
            name="Path" if attractive else "Path: " + str(path),
            legendgroup="Path: " + str(path) if attractive else None,
            legendgrouptitle_text="Path: " + str(path) if attractive else None,
            showlegend=True,
            legend=legend,
            hoverinfo="none",
            visible='legendonly',
        ))

    return paths_lines


def get_paths_population_cells(population_coords: np.ndarray, population_cells_paths: dict) -> list[go.Scatter]:
    """
    Returns a list of Scatter objects, each representing the population cells near the origin (an airport node) of each
    path.

    Args:
        population_coords (np.ndarray): A NumPy array of shape (num_population_cells, 2) containing (x, y) coordinates
            of population cell centers.
        population_cells_paths (dict): A dictionary mapping each population cell index to a list of paths (each path is
            a list of node IDs) starting from an airport near that population cell.

    Returns:
        list: A list of Scatter objects, each representing the population cells near the origin (an airport node) of
            each path.
    """
    paths_origins_population_cells = []
    added_paths = []
    for pop_cell, paths in population_cells_paths.items():
        for path in paths:
            if path not in added_paths:
                added_paths.append(path)
                show = True
            else:
                show = False
            paths_origins_population_cells.append(go.Scatter(
                x=[population_coords[pop_cell][0]], y=[population_coords[pop_cell][1]],
                mode='markers',
                marker=dict(
                    color='rgba(147, 112, 219, 0.5)',
                    size=8,
                    line=dict(
                        color='rgba(147, 112, 219, 1)',
                        width=1
                    ),
                    symbol="diamond"
                ),
                name="Path Origin Population Cells",
                hovertemplate="Population Cell: {}<br>".format(pop_cell) +
                              "Coordinates: %{x:.1f}km x %{y:.1f}km <extra></extra>",
                showlegend=show,
                legend="legend4",
                legendgroup="Path: " + str(path),
                visible="legendonly",
            ))

    return paths_origins_population_cells


def get_ground_dist_paths_origins_airports(airports_coords: np.ndarray, paths: np.ndarray,
                                           max_ground_distance: float) -> list[go.Scatter]:
    """
    Returns a list of Scatter objects representing the circular area of 'max ground distance' from each path origin (an
    airport node).

    Args:
        airports_coords (np.ndarray): A NumPy array of shape (num_airports, 2) containing (x, y) coordinates of each
            airport.
        paths (np.ndarray): A NumPy array of paths (each path is a list of node IDs).
        max_ground_distance (float): Maximum allowed ground distance to consider a population cell "near" an airport.

    Returns:
        list: A list of Scatter objects representing the circular area of 'max ground distance' from each path origin
        (an airport node).
    """

    circle_traces = []
    for i, path in enumerate(paths):
        last_airport = path[0]
        x_center, y_center = airports_coords[last_airport]
        theta = np.linspace(0, 2 * np.pi, 200)
        x_circle = x_center + max_ground_distance * np.cos(theta)
        y_circle = y_center + max_ground_distance * np.sin(theta)
        circle_trace = go.Scatter(
            x=x_circle,
            y=y_circle,
            mode='lines',
            line=dict(color='rgba(147, 112, 219, 0.5)', width=2, dash='dot'),
            name="Max Ground Distance From Airport {}".format(last_airport),
            hovertemplate="Airport: {}<br>".format(last_airport) +
                          "Max Ground Distance: {}km<extra></extra>".format(max_ground_distance),
            showlegend=True,
            legend="legend4",
            legendgroup="Path: " + str(path),
            visible="legendonly",
        )
        circle_traces.append(circle_trace)

    return circle_traces


def plot_dataset_and_solution(population_coords: np.ndarray, population_density: np.ndarray,
                              airports_coords: np.ndarray,
                              airport_distances: dict, graph_below_tau: nx.Graph, graph_above_tau: nx.Graph,
                              destination_airports: np.ndarray, destination_cells: list, max_ground_distance: float,
                              min_distance_to_destination_cells: float, all_paths: np.ndarray,
                              attractive_paths: np.ndarray,
                              population_cells_paths: dict, charging_airports: list, active_path_indices: np.ndarray,
                              plot_name: str,
                              simple_plot_enable: bool, save_plot: bool) -> None:
    """
    Main function to plot the full dataset and the solution of the EANC-REG model.

    Args:
        population_coords (np.ndarray): A NumPy array of shape (num_population_cells, 2) containing (x, y) coordinates
            of population cell centers.
        population_density (np.ndarray): A NumPy array of population density values (integers) with the same length as
            `population_coords`.
        airports_coords (np.ndarray): A NumPy array of shape (num_airports, 2) containing (x, y) coordinates of each
            airport.
        airport_distances (dict): A dictionary where each key is a tuple (i, j) representing a pair of node indices, 7
            and the value is the Euclidean distance between node i and node j.
        graph_below_tau (nx.Graph): A NetworkX graph containing the edges below the distance threshold tau.
        graph_above_tau (nx.Graph): A NetworkX graph containing the edges above the distance threshold tau.
        destination_airports (np.ndarray): A NumPy array of destination airports indices close to the destination cells.
        destination_cells (list): An array of destination cell indices.
        max_ground_distance (float): Maximum allowed ground distance to consider a population cell "near" an airport.
        min_distance_to_destination_cells(float): Minimum allowed distance to consider a population cell or an airport
            from the destination cells (km).
        all_paths (np.ndarray): A NumPy array of all paths (each path is a list of node IDs).
        attractive_paths (np.ndarray):  A NumPy array of attractive paths (each path is a list of node IDs).
        population_cells_paths (dict):  A dictionary mapping each population cell index to a list of paths (each path is
            a list of node IDs) starting from an airport near that population cell.
        charging_airports (list): A list of airports indices representing the charging bases.
        active_path_indices (np.ndarray): A NumPy array of active paths indices.
        plot_name (str): A string name for the plot.
        simple_plot_enable (bool): True to plot minimum information.
        save_plot (bool): Save plot in a html file.

    Return:
        none
    """
    fig = go.Figure()

    for grid in get_population_grid(population_coords=population_coords):
        fig.add_shape(grid)

    density_heatmap = get_population_heatmap(population_coords=population_coords, population_density=population_density)
    fig.add_trace(density_heatmap)

    population_cell_markers = get_population_cells(population_coords=population_coords)
    fig.add_trace(population_cell_markers)

    destination_cell_markers = get_destination_cells(population_coords=population_coords,
                                                     destination_cells=destination_cells)
    fig.add_trace(destination_cell_markers)

    round_dist_destination_cells = get_max_ground_dist_destination_cells(population_coords=population_coords,
                                                                         destination_cells=destination_cells,
                                                                         max_ground_distance=max_ground_distance)
    fig.add_traces(round_dist_destination_cells)

    round_dist_destination_cells = get_min_dist_destination_cells(population_coords=population_coords,
                                                                  destination_cells=destination_cells,
                                                                  min_distance_to_destination_cells=min_distance_to_destination_cells)
    fig.add_traces(round_dist_destination_cells)

    if not simple_plot_enable:
        connections_above_tau = get_connections(airports_coords=airports_coords, airport_distances=airport_distances,
                                                graph=graph_above_tau)
        fig.add_traces(connections_above_tau)

        connections_below_tau = get_connections(airports_coords=airports_coords, airport_distances=airport_distances,
                                                graph=graph_below_tau, above=False)
        fig.add_traces(connections_below_tau)

        paths = get_paths(airports_coords=airports_coords, paths=all_paths, attractive=False)
        fig.add_traces(paths)

    paths = get_paths(airports_coords=airports_coords,
                      paths=np.array([path for i, path in enumerate(attractive_paths)
                                      if i not in list(active_path_indices)], dtype=object))
    fig.add_traces(paths)

    active = [len(fig.data) + i for i in range(len(active_path_indices))]
    paths = get_paths(airports_coords=airports_coords,
                      paths=np.array([path for i, path in enumerate(attractive_paths)
                                      if i in list(active_path_indices)], dtype=object))
    fig.add_traces(paths)

    path_origins_population_cells = get_paths_population_cells(population_coords=population_coords,
                                                               population_cells_paths=population_cells_paths)
    active_legend_group = [path.legendgroup for path in paths]

    for i, cell in enumerate(path_origins_population_cells):
        if cell.legendgroup in active_legend_group:
            active.append(len(fig.data) + i)

    fig.add_traces(path_origins_population_cells)

    if not simple_plot_enable:
        ground_dist_paths_origins_airports = (
            get_ground_dist_paths_origins_airports(airports_coords=airports_coords, paths=attractive_paths,
                                                   max_ground_distance=max_ground_distance))
        fig.add_traces(ground_dist_paths_origins_airports)

    charging_airport_icons = get_charging_airport_icons(airports_coords=airports_coords,
                                                        charging_airports=charging_airports)
    fig.add_trace(charging_airport_icons)

    airport_icons = get_airport_icons(airports_coords=airports_coords)
    fig.add_trace(airport_icons)

    airport_markers = get_airports(airports_coords=airports_coords)
    fig.add_trace(airport_markers)

    destination_airport_markers = get_destination_airports(airports_coords=airports_coords,
                                                           destination_airports=destination_airports)
    fig.add_trace(destination_airport_markers)

    airport_charger = get_charging_airports(airports_coords=airports_coords, charging_airports=charging_airports)
    fig.add_trace(airport_charger)

    cell_size = abs(float(population_coords[1][0] - population_coords[0][0]))
    y_range = [-cell_size / 2, max(population_coords[:, 1]) + cell_size]
    x_range = [-cell_size / 2, max(population_coords[:, 0]) + cell_size]
    visibility = [i.visible for i in fig.data]

    fig.update_layout(
        margin=dict(b=100),
        title=dict(
            text="EACN REG",
            x=0,
            y=0.98,
            font=dict(
                size=20,
                weight="bold"
            ),
            xanchor="left",
            yanchor="top",
            xref="paper",
            subtitle=dict(
                text="Electric aircraft charging network design for regional routes {}".format(plot_name),
                font=dict(
                    size=13,
                ),
            )
        ),
        legend=dict(
            title=dict(
                text="Dataset:",
                font=dict(
                    size=13,
                    weight="bold"
                ),
            ),
            xref="container",
            yref="container",
            xanchor="left",
            x=0.85,
            y=0.9,
        ),
        legend2=dict(
            title=dict(
                text="Connections:",
                font=dict(
                    size=13,
                    weight="bold"
                ),
            ),
            xref="container",
            yref="container",
            xanchor="left",
            x=0.85,
            y=0.7,
        ),
        legend3=dict(
            title=dict(
                text="Paths:",
                font=dict(
                    size=13,
                    weight="bold"
                ),
            ),
            xref="container",
            yref="container",
            maxheight=0.5,
            xanchor="left",
            yanchor="top",
            x=0.85,
            y=0.6,
        ),
        legend4=dict(
            title=dict(
                text="Attractive Paths:",
                font=dict(
                    size=13,
                    weight="bold"
                ),
            ),
            xref="container",
            yref="container",
            maxheight=0.5,
            xanchor="left",
            yanchor="top",
            x=0.85,
            y=0.6,
        ),
        legend5=dict(
            title=dict(
                text="Solutions:",
                font=dict(
                    size=13,
                    weight="bold"
                ),
            ),
            xref="container",
            yref="container",
            xanchor="left",
            yanchor="top",
            x=0.85,
            y=0.09,
        ),

        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            ticksuffix="km",
            range=x_range,

        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            ticksuffix="km",
            range=y_range
        ),
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        label="Show Heatmap",
                        method="update",
                        args=[{"visible": get_visibility([0], visibility, True)}]
                    ),
                    dict(
                        label="Hide Heatmap",
                        method="update",
                        args=[{"visible": get_visibility([0], visibility, False)}],
                    )
                ],
                direction="down",
                showactive=True,
                x=1,
                y=1.02,
                xanchor="right",
                yanchor="bottom",
                pad={"r": 5, "t": 5}
            ),
            dict(
                buttons=[
                    dict(
                        label="Attractive Paths",
                        method="update",
                        args=[{"legend3.visible": False}, {"legend4.visible": True}],
                    ),
                    dict(
                        label="All Paths",
                        method="update",
                        args=[{"legend3.visible": True}, {"legend4.visible": False}],
                    ),
                ],
                direction="down",
                showactive=True,
                x=0.9,
                y=1.02,
                xanchor="right",
                yanchor="bottom",
                pad={"r": 5, "t": 5}
            ),
            dict(
                buttons=[
                    dict(
                        label="Hide Active Paths",
                        method="update",
                        args=[{"visible": get_visibility(active, visibility, 'legendonly')}],
                    ),
                    dict(
                        label="Show Active Paths",
                        method="update",
                        args=[{"visible": get_visibility(active, visibility, True)}],
                    ),
                ],
                direction="down",
                showactive=True,
                x=0.8,
                y=1.02,
                xanchor="right",
                yanchor="bottom",
                pad={"r": 5, "t": 5}
            )
        ]
    )
    fig.layout.plot_bgcolor = 'rgb(250, 250, 250)'
    if save_plot:
        fig.write_html("{}_{}.html".format(plot_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    else:
        fig.show()


def get_visibility(indices, visibility, status) -> list:
    """
    Helper to control visibility of the data in the plot
    """
    for idx in indices:
        visibility[idx] = status
    return [ii for ii in visibility]
