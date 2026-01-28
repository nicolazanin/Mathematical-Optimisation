import datetime
import pandas as pd
import logging
import geopandas as gpd
from shapely.geometry import Point, box
import plotly.express as px
import numpy as np
from pyproj import Transformer
import plotly.graph_objects as go

from utils.settings import settings

_logger = logging.getLogger(__name__)
_logger.setLevel(settings.logging_lvl)

transformer2xy = Transformer.from_crs("EPSG:4326", "EPSG:3006", always_xy=True)
transformer2latlon = Transformer.from_crs("EPSG:3006", "EPSG:4326", always_xy=True)


class Airport:
    """
    Represents an airport and its main characteristics.

    Attributes:
        city_served (str): City served by the airport.
        icao (str): ICAO airport code.
        iata (str | None): IATA airport code (may be None).
        airport_name (str): Full name of the airport.
        usage (str): Airport usage (e.g. Public, Private).
        runways (list): List of runways information parsed from the CSV.
        lat (float): Latitude of the airport.
        lon (float): Longitude of the airport.
        passengers (int | None): Annual passenger count.
        curr_active (bool): True if passenger data is available, otherwise False.
        activation_cost (int): Activation cost of the airport as charging base.
        x (float): x coordinate of the airport in km.
        y (float): y coordinate of the airport in km.
    """

    def __init__(self, city_served, icao, iata, airport_name, usage, runways, lat, lon, passengers):
        """
        Initializes an Airport object.

        Args:
            city_served (str): City served by the airport.
            icao (str): ICAO airport code.
            iata (str | None): IATA airport code.
            airport_name (str): Name of the airport.
            usage (str): Usage type of the airport.
            runways (str): Runway data as a string from the CSV file.
            lat (float): Latitude.
            lon (float): Longitude.
            passengers (int | None): Annual passenger count.
        """
        self.city_served = city_served
        self.icao = icao
        self.iata = iata
        self.airport_name = airport_name
        self.usage = usage
        self.runways = parse_runways(runways)
        self.lat = lat
        self.lon = lon
        self.passengers = passengers if passengers is not None else 0
        self.curr_active = True if passengers else False
        self.activation_cost = 0
        x_m, y_m = transformer2xy.transform(lon, lat)
        self.x, self.y = x_m / 1000, y_m / 1000

    def __str__(self):
        return f"Airport Name: {self.airport_name}<br>ICAO: {self.icao}<br>IATA: {self.iata}<br>" \
               f"Runways: {self.runways}<br>Location: ({self.lat:.3f}°N, {self.lon:.3f}°E)<br>Annual Passengers: {self.passengers:.0f}"


def parse_runways(runways: str) -> list:
    """
    Helper method to parse runway information into a more usable format.

    Args:
        runways (str): A string for runways information structured as ((direction, length, runway surface), ...).

    Returns:
        list: A list of information for each runway.
    """
    runways_list = []
    if runways:
        runway_details = runways.split('),')
        for runway in runway_details:
            runways_list.append(runway.replace('(', '').replace(')', ''))

    return runways_list


def get_airports(airports_file: str, only_active: bool) -> np.ndarray:
    """
    Generates a list of Airports from airports list .csv file.

    Args:
        airports_file (str): Path to the airports list .csv file.
        only_active (bool): True if only active airports will be returned.

    Returns:
        np.ndarray: A NumPy array of Airports
    """
    airports = []
    df_airports = pd.read_csv(filepath_or_buffer=airports_file, sep=";").replace(np.nan, None)
    for cos in df_airports.values:
        if cos[8] is not None and only_active:
            airports.append(Airport(*cos))
        elif not only_active:
            airports.append(Airport(*cos))

    _logger.info("Retrieved {} airports from airports list file".format(len(airports)))
    return np.array(airports)


def get_population_cells(df_population: pd.DataFrame, cell_area: float) -> gpd.GeoDataFrame:
    """
    From a population density file of Sweden generates a grid of square cells and computes the center coordinates of
    each cell. Creates a GeoDataFrame containing a grid of square cells and the population density for each cell.

    Args:
        df_population (pd.DataFrame): A Dataframe containing population information.
        cell_area (float): Area of square cell. Assumes square cells (width = height = sqrt(area)).

    Returns:
        gpd.DataFrame: A GeoDataFrame containing a grid of square cells and the population density for each cell.
    """
    cell_width = cell_height = np.sqrt(cell_area)
    delta = 30 / 3600
    df_population["cell_area_km2"] = (111.32 * delta * 111.32 * delta * np.cos(np.deg2rad(df_population["Y"])))
    df_population["population"] = df_population["Z"] * df_population["cell_area_km2"]

    gdf = gpd.GeoDataFrame(df_population, geometry=gpd.points_from_xy(df_population.X, df_population.Y),
                           crs="EPSG:4326")  # WGS84 we load the data in lat, lon

    gdf = gdf.to_crs("EPSG:3006")  # Project to metric CRS (meters) Sweden: SWEREF99 TM

    xmin, ymin, xmax, ymax = gdf.total_bounds

    grid_cells = []
    x_coords = np.arange(xmin, xmax, cell_width * 1000)
    y_coords = np.arange(ymin, ymax, cell_height * 1000)

    for x in x_coords:
        for y in y_coords:
            grid_cells.append(box(x - cell_width * 1000 / 2, y - cell_height * 1000 / 2,
                                  x + cell_width * 1000 / 2, y + cell_height * 1000 / 2))

    population_cells = gpd.GeoDataFrame({"geometry": grid_cells}, crs="EPSG:3006")
    joined = gpd.sjoin(gdf, population_cells, predicate="within")
    population_cells["population"] = joined.groupby("index_right")["population"].sum()
    population_cells["population"] = population_cells["population"].fillna(0)
    population_cells["density"] = population_cells["population"] / (
                population_cells.geometry.area / 1_000_000)  # (people per km²)
    population_cells = population_cells[population_cells["density"] > 0.05]
    population_cells = population_cells.reset_index(drop=True)
    _logger.info("Generated a grid of {} cells (cell area={}km^2) form population dataframe".
                 format(len(population_cells), cell_area))

    return population_cells


def get_population_cells_centroids(population_cells: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Returns center points of population grid cells.

    Args:
        population_cells (gpd.GeoDataFrame): A GeoDataFrame containing a grid of square cells and the population density
            for each cell.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the centroids of the grid cells.
    """

    centers = population_cells.copy()
    centers.geometry = centers.geometry.centroid

    centers["x"] = centers.geometry.x / 1000
    centers["y"] = centers.geometry.y / 1000

    return centers


def plot_case_study(population_cells: gpd.GeoDataFrame, population_cells_centroids: gpd.GeoDataFrame,
                    airports: np.ndarray, destination_airports: np.ndarray, destination_cells: list, edges: list,
                    charging_airports: list, population_covered: list, plot_name: str, save_plot: bool):
    """
    Returns center points of population grid cells.

    Args:
        population_cells (gpd.GeoDataFrame): A GeoDataFrame containing a grid of square cells and the population density
            for each cell.
        population_cells_centroids (gpd.GeoDataFrame): A GeoDataFrame containing the centroids of the grid cells
        airports (np.ndarray): A NumPy array of Airports
        destination_airports (np.ndarray): A NumPy array of destination airports indices close to the destination cells.
        destination_cells (list): An array of destination cell indices.
        edges (list): A list of feasible edges.
        charging_airports (list): A list of airports indices representing the charging bases.
        population_covered (list): A list of population covered indices
        plot_name (str): A string name for the plot.
        save_plot (bool): Save plot in a html file.

    Returns:
        none
    """
    population_cells = population_cells.to_crs("EPSG:4326")
    population_cells_centroids = population_cells_centroids.to_crs("EPSG:4326")

    custom_colorscale = [
        [0.0, 'rgba(220,30,0,0)'],
        [0.25, 'rgba(225,30,0,0.3)'],
        [0.5, 'rgba(230,30,0,0.4)'],
        [0.75, 'rgba(235,30,0,0.45)'],
        [1.0, 'rgba(240,30,0,0.5)']
    ]
    fig = px.choropleth_map(
        population_cells,
        geojson=population_cells.geometry,
        locations=population_cells.index,
        color="density",
        color_continuous_scale=custom_colorscale,
        center={"lat": 63.0, "lon": 15.0},
        zoom=4.8,
        opacity=0.4,
        labels={"density": "People per km²"},
        title="Population Density in Sweden (50 km grid)"
    )

    labels = np.array([str(num) for num in range(len(population_cells_centroids))])
    fig.add_trace(go.Scattermap(
        lon=population_cells_centroids.geometry.x,
        lat=population_cells_centroids.geometry.y,
        mode='markers',
        marker=go.scattermap.Marker(
            color='rgb(128, 128, 128)',
            size=3,
            opacity=0.5,
        ),
        name='Population Cells',
        customdata=labels,
        hovertemplate="Population Cell: %{customdata}<br>"
                      "Coordinates: %{lat:.3f}°N, %{lon:.3f}°E <extra></extra>",
        showlegend=True,
        legendrank=1
    ))
    fig.add_trace(go.Scattermap(
        lon=population_cells_centroids.geometry.values.x[population_covered],
        lat=population_cells_centroids.geometry.values.y[population_covered],
        mode='markers',
        marker=dict(
            color='rgb(147, 112, 219)',
            size=5,
        ),
        name='Population Cells Covered',
        customdata=labels[population_covered],
        hovertemplate="Population Cell: %{customdata}<br>"
                      "Coordinates: %{lat:.3f}°N, %{lon:.3f}°E <extra></extra>",
        showlegend=True,
        legendrank=7
    ))

    airports_lon = np.array([])
    airports_lat = np.array([])
    airports_info = np.array([])
    for i, airport in enumerate(airports):
        airports_lon = np.append(airports_lon, airport.lon)
        airports_lat = np.append(airports_lat, airport.lat)
        airports_info = np.append(airports_info, str(airport))

    for i, edge in enumerate(edges):
        lat, lon = great_circle_path(airports_lat[edge[0]], airports_lon[edge[0]], airports_lat[edge[1]],
                                     airports_lon[edge[1]], num_points=20)
        fig.add_trace(go.Scattermap(
            mode="lines",
            lon=lon,
            lat=lat,
            line=dict(
                color="rgba(0, 0, 0, 1)",
                width=1.2
            ),
            name="Edges",
            legendgroup="opt_edges",
            showlegend=(i == 0),
            hoverinfo="none",
            legendrank=8
        ))

    labels = [str(num) for num in range(len(destination_cells))]
    fig.add_trace(go.Scattermap(
        lon=population_cells_centroids.geometry.values.x[destination_cells],
        lat=population_cells_centroids.geometry.values.y[destination_cells],
        mode='markers',
        marker=go.scattermap.Marker(
            color='rgb(135, 206, 250)',
            size=13,
            opacity=1,
        ),
        name="Destination Cells",
        customdata=labels,
        hovertemplate="Destination Cell: %{customdata} <extra></extra>",
        showlegend=True,
        legendrank=2
    ))

    airports_info = np.array([airports_info, [i for i in range(len(airports_lon))]]).T
    fig.add_trace(go.Scattermap(
        lon=airports_lon,
        lat=airports_lat,
        mode='markers',
        marker=dict(
            color='rgb(0, 0, 0)',
            size=8,
            symbol='airport'
        ),
        customdata=airports_info,
        hovertemplate="Airport: %{customdata[1]}<br>%{customdata[0]} <extra></extra>",
        name="Airports " + "✈️",
        legendgroup="airports",
        showlegend=True,
        legendrank=3
    ))

    fig.add_trace(go.Scattermap(
        lon=airports_lon[destination_airports],
        lat=airports_lat[destination_airports],
        mode='markers',
        marker=go.scattermap.Marker(
            color='rgb(250, 154, 255)',
            size=15,
            opacity=0.5,
        ),
        name="Destination Airports " + "✈️",
        customdata=airports_info[destination_airports],
        hovertemplate="Airport: %{customdata[1]}<br>%{customdata[0]} <extra></extra>",
        showlegend=True,
        legendrank=4
    ))

    fig.add_trace(go.Scattermap(
        lon=airports_lon[charging_airports],
        lat=airports_lat[charging_airports],
        mode='markers',
        marker=dict(
            color='rgb(141, 204, 71)',
            size=15,
            opacity=0.5,
        ),
        customdata=airports_info,
        hovertemplate="Airport: %{customdata[1]}<br>%{customdata[0]} <extra></extra>",
        name="Charging Airports " + "🔋",
        showlegend=True,
        legendrank=9
    ))

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
        coloraxis_colorbar=dict(
            title="Population Density [people/km²]: ",
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.03,
            yanchor="top",
            thickness=10,
            len=0.5
        ),
        legend=dict(
            title=dict(
                text="Dataset and solutions:",
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
    )
    if save_plot:
        fig.write_html("{}_{}.html".format(plot_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    else:
        fig.show()
    fig.show()


def deg2rad(degrees):
    return np.deg2rad(degrees)


def great_circle_path(lat1, lon1, lat2, lon2, num_points=10):
    """
    Compute intermediate points along a great-circle path between two coordinates.
    (uses spherical linear interpolation)

    Args:
        lat1 (float): Latitude of the starting point in degrees.
        lon1 (float): Longitude of the starting point in degrees.
        lat2 (float): Latitude of the ending point in degrees.
        lon2 (float): Longitude of the ending point in degrees.
        num_points (int, optional): Number of intermediate points between
            the start and end locations. Defaults to 10.

    Returns:
        tuple: A tuple containing two lists for  latitudes and longitudes (in degrees) of the great-circle path,
            including the start and end points.
    """
    lat1, lon1, lat2, lon2 = map(deg2rad, [lat1, lon1, lat2, lon2])
    delta_lon = lon2 - lon1
    central_angle = np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(delta_lon))

    lat = []
    lon = []
    for i in range(num_points + 2):  # +2 because we want start and end points too
        t = i / (num_points + 1)

        A = np.sin((1 - t) * central_angle) / np.sin(central_angle)
        B = np.sin(t * central_angle) / np.sin(central_angle)

        x = A * np.cos(lat1) * np.cos(lon1) + B * np.cos(lat2) * np.cos(lon2)
        y = A * np.cos(lat1) * np.sin(lon1) + B * np.cos(lat2) * np.sin(lon2)
        z = A * np.sin(lat1) + B * np.sin(lat2)
        interpolated_lat = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
        interpolated_lon = np.arctan2(y, x)
        lat.append(np.rad2deg(interpolated_lat))
        lon.append(np.rad2deg(interpolated_lon))

    return lat, lon
