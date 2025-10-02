import plotly.graph_objects as go
import numpy as np
import plotly.io as pio

pio.renderers.default = "browser"


def get_population_heatmap(population_coords, population_density) -> go.Heatmap:
    # Convert to NumPy arrays for easy reshaping
    population_coords = np.array(population_coords)
    density = np.array(population_density)

    # Determine grid size (assumes coordinates are sorted row-major)
    x_unique = np.unique(population_coords[:, 0])
    y_unique = np.unique(population_coords[:, 1])

    grid_width = len(x_unique)
    grid_height = len(y_unique)

    # Reshape into 2D
    z_grid = density.reshape((grid_height, grid_width))
    custom_colorscale = [
        [0.0, 'rgba(220,30,0,0)'],  # white
        [0.25, 'rgba(225,30,0,0.15)'],  # light blue
        [0.5, 'rgba(230,30,0,0.2)'],  # medium blue
        [0.75, 'rgba(235,30,0,0.3)'],  # deep blue
        [1.0, 'rgba(240,30,0,0.4)']  # darkest blue
    ]

    # Create heatmap
    density_heatmap = go.Heatmap(
        z=z_grid,
        x=x_unique,
        y=np.sort(y_unique),  # sorted y for proper axis display
        colorscale=custom_colorscale,
        colorbar=dict(title=dict(
            text="Population Density: ",
            font=dict(
                size=12,
            ),
        ), orientation='h',
            x=0.5,  # center horizontally
            y=-0.03,  # below the plot, adjust if needed
            xanchor='center',
            yanchor='top',
            thickness=10,  # height of colorbar
            len=0.5
        ),
        zmin=density.min(),
        zmax=density.max(),
        name="Population Cell",
        hovertemplate="Population density: %{z} people/km² <extra></extra>",
        legend="legend1",
    )
    return density_heatmap


def get_population_cells(population_coords) -> go.Scatter:
    labels = [str(num) for num in range(len(population_coords))]
    population_cell_markers = go.Scatter(
        x=population_coords[:, 0],
        y=population_coords[:, 1],
        mode='markers',
        marker=dict(size=5, color='rgba(128, 128, 128, 0.5)'),
        name='Population Cell',
        customdata=labels,
        showlegend=True,
        hovertemplate="Population Cell: %{customdata}<br>"
                      "Coordinates: %{x:.1f}km x %{y:.1f}km <extra></extra>",
        legend="legend1",
        legendrank=1
    )
    return population_cell_markers


def get_destination_cells(population_coords, destination_cells) -> go.Scatter:
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
        name="Destination Cell",
        customdata=labels,
        hovertemplate="Destination Cell: %{customdata} <extra></extra>",
        showlegend=True,
        legend="legend1",
        legendrank=2
    )
    return destination_cell_markers


def get_ground_dist_destination_cells(population_coords, destination_cells, max_ground_distance) -> list:
    """
    Draws circular areas with radius = max_ground_distance (in data units) around destination cells.
    Returns a list of Scatter traces.
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


def get_population_grid(population_coords):
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


def get_airports(airports_coords):
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
        legendrank=4
    )
    return airport_markers


def get_airport_icons(airports_coords):
    emoji = "✈️"
    airport_icons = go.Scatter(
        x=airports_coords[:, 0],
        y=airports_coords[:, 1],
        mode='text',
        text=[emoji] * len(airports_coords[:, 0]),
        textfont=dict(
            size=25,
        ),
        name="Airports",
        legendgroup="airports",
        showlegend=False,
        legend="legend1"
    )
    return airport_icons


def get_airport_chargers(airports_coords, charging_stations):
    labels = [str(num) for num in charging_stations]
    airport_markers = go.Scatter(
        x=airports_coords[charging_stations, 0],
        y=airports_coords[charging_stations, 1],
        mode='markers',
        marker=dict(
            color='rgba(76, 235, 52, 1)',
            size=10,
            line=dict(
                color='rgb(76, 235, 52)',
                width=3
            )
        ),
        name="Charging Airports " + "🔋",
        legendgroup="charging_stations",
        showlegend=True,
        visible='legendonly',
        customdata=labels,
        hovertemplate="Charging Airport: %{customdata}<extra></extra>",
        legend="legend5",
    )
    return airport_markers


def get_airport_charger_icons(airports_coords, charging_stations):
    emoji = "🔋"
    airport_charger_icons = go.Scatter(
        x=airports_coords[charging_stations, 0] - 20,
        y=airports_coords[charging_stations, 1],
        mode='text',
        text=[emoji] * len(airports_coords[charging_stations, 0]),
        textfont=dict(
            size=30,
        ),
        name="Charging Stations",
        showlegend=False,
        legend="legend5",
        legendgroup="charging_stations",
        visible='legendonly',
        hoverinfo="none",
    )
    return airport_charger_icons


def get_destination_airports(airports_coords, destination_airports):
    labels = np.array([[str(num) for num in range(len(destination_airports))],
                       [str(num) for num in destination_airports]]).T
    destination_airport_markers = go.Scatter(
        x=airports_coords[destination_airports, 0],
        y=airports_coords[destination_airports, 1],
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
        legendrank=5
    )
    return destination_airport_markers


def get_connections(airports_coords, airport_distances, graph, above=True):
    """
    Plot connections between airports (short and long range).
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


def get_paths(airports_coords, paths, attractive=True):
    """
    Plot connections between airports (short and long range).
    """
    paths_lines = []
    if attractive:
        legend = "legend4"
        visible = 'legendonly'
    else:
        legend = "legend3"
        visible = 'legendonly'
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
            line=dict(width=4),
            name="Path" if attractive else "Path: " + str(path),
            legendgroup="Path: " + str(path) if attractive else None,
            legendgrouptitle_text="Path: " + str(path) if attractive else None,
            showlegend=True,
            legend=legend,
            hoverinfo="none",
            visible=visible,
        ))
    return paths_lines


def get_paths_origins_population_cells(population_coords, paths, population_cells_near_airports):
    """
    Plot connections between airports (short and long range).
    """
    paths_origins_population_cells = []
    for i, path in enumerate(paths):
        last_airport = path[0]
        population_cells = population_cells_near_airports[last_airport]
        for j, cell in enumerate(population_cells):
            paths_origins_population_cells.append(go.Scatter(
                x=[population_coords[cell][0]], y=[population_coords[cell][1]],
                mode='markers',
                marker=dict(
                    color='rgba(147, 112, 219, 0.5)',
                    size=8,
                    line=dict(
                        color='rgba(147, 112, 219, 1)',
                        width=1
                    )
                ),
                marker_symbol="diamond",
                name="Path Origin Population Cells",
                hovertemplate="Population Cell: {}<br>".format(cell) +
                              "Coordinates: %{x:.1f}km x %{y:.1f}km <extra></extra>",
                showlegend=(j == 0),
                legend="legend4",
                legendgroup="Path: " + str(path),
                visible="legendonly",
            ))
    return paths_origins_population_cells


def get_ground_dist_paths_origins_airports(airports_coords, paths, max_ground_distance) -> list:
    """
    Draws circular areas with radius = max_ground_distance (in data units) around destination cells.
    Returns a list of Scatter traces.
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


def plot_dataset(population_coords, population_density, airports_coords,
                 airport_distances, graph_below_tau, graph_above_tau, destination_airports,
                 destination_cells, max_ground_distance, all_simple_paths, attractive_simple_paths,
                 population_cells_near_airports, active_bases):
    """
    Main function to plot the full dataset in Plotly.
    """
    fig = go.Figure()

    population_grid = get_population_grid(population_coords)
    for grid in population_grid:
        fig.add_shape(grid)

    density_heatmap = get_population_heatmap(population_coords, population_density)
    fig.add_trace(density_heatmap)

    population_cell_markers = get_population_cells(population_coords)
    fig.add_trace(population_cell_markers)

    destination_cell_markers = get_destination_cells(population_coords, destination_cells)
    fig.add_trace(destination_cell_markers)

    round_dist_destination_cells = get_ground_dist_destination_cells(population_coords, destination_cells,
                                                                     max_ground_distance)
    fig.add_traces(round_dist_destination_cells)

    connections_above_tau = get_connections(airports_coords, airport_distances, graph_above_tau)
    fig.add_traces(connections_above_tau)

    connections_below_tau = get_connections(airports_coords, airport_distances, graph_below_tau, above=False)
    fig.add_traces(connections_below_tau)

    paths = get_paths(airports_coords, all_simple_paths, attractive=False)
    fig.add_traces(paths)

    paths = get_paths(airports_coords, attractive_simple_paths)
    fig.add_traces(paths)

    path_origins_population_cells = get_paths_origins_population_cells(population_coords, attractive_simple_paths,
                                                                       population_cells_near_airports)
    fig.add_traces(path_origins_population_cells)

    ground_dist_paths_origins_airports = get_ground_dist_paths_origins_airports(airports_coords,
                                                                                attractive_simple_paths,
                                                                                max_ground_distance)
    fig.add_traces(ground_dist_paths_origins_airports)

    airport_charger_icons = get_airport_charger_icons(airports_coords, active_bases)
    fig.add_trace(airport_charger_icons)

    airport_icons = get_airport_icons(airports_coords)
    fig.add_trace(airport_icons)

    airport_markers = get_airports(airports_coords)
    fig.add_trace(airport_markers)

    destination_airport_markers = get_destination_airports(airports_coords, destination_airports)
    fig.add_trace(destination_airport_markers)

    airport_charger = get_airport_chargers(airports_coords, active_bases)
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
                text="Electric aircraft charging network design for regional routes",
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
            y=0.73,
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
            x=0.85,
            y=0.15,
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
            x=0.85,
            y=0.15,
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
                        args=[{"visible": get_visibility(0, visibility, True)}, ]
                    ),
                    dict(
                        label="Hide Heatmap",
                        method="update",
                        args=[{"visible": get_visibility(0, visibility, False)}],
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
            )
        ]
    )
    fig.show()


def get_visibility(idx, visibility, status):
    visibility[idx] = status
    return [ii for ii in visibility]
