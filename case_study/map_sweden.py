import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import plotly.express as px
import numpy as np
import plotly.graph_objects as go


# Define a function to convert degrees to radians
def deg2rad(degrees):
    return np.deg2rad(degrees)


# Function to compute the intermediate points along the great circle
def great_circle_path(lat1, lon1, lat2, lon2, num_points=10):
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(deg2rad, [lat1, lon1, lat2, lon2])

    # Calculate the central angle between the two points using the spherical law of cosines
    delta_lon = lon2 - lon1
    central_angle = np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(delta_lon))

    # Generate intermediate points
    lat = []
    lon = []
    for i in range(num_points + 2):  # +2 because we want start and end points too
        # Compute interpolation parameter (t) between 0 and 1
        t = i / (num_points + 1)

        # Spherical linear interpolation (SLERP)
        A = np.sin((1 - t) * central_angle) / np.sin(central_angle)
        B = np.sin(t * central_angle) / np.sin(central_angle)

        # Calculate the interpolated latitude and longitude
        x = A * np.cos(lat1) * np.cos(lon1) + B * np.cos(lat2) * np.cos(lon2)
        y = A * np.cos(lat1) * np.sin(lon1) + B * np.cos(lat2) * np.sin(lon2)
        z = A * np.sin(lat1) + B * np.sin(lat2)

        # Convert the interpolated Cartesian coordinates back to latitude and longitude
        interpolated_lat = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
        interpolated_lon = np.arctan2(y, x)

        # Convert radians back to degrees
        lat.append(np.rad2deg(interpolated_lat))
        lon.append(np.rad2deg(interpolated_lon))

    return lat, lon

df = pd.read_csv("swe_pd_2019_1km_ASCII_XYZ.csv") # from https://hub.worldpop.org/geodata/summary?id=44035

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.X, df.Y),
    crs="EPSG:4326"  # WGS84 we load the data in lat, lon
)

gdf = gdf.to_crs("EPSG:3006") # Project to metric CRS (meters) Sweden: SWEREF99 TM

grid_size = 50_000

xmin, ymin, xmax, ymax = gdf.total_bounds

grid_cells = []
x_coords = np.arange(xmin, xmax, grid_size)
y_coords = np.arange(ymin, ymax, grid_size)

for x in x_coords:
    for y in y_coords:
        grid_cells.append(box(x - grid_size/2, y- grid_size/2, x + grid_size/2, y + grid_size/2))

grid = gpd.GeoDataFrame(
    {"geometry": grid_cells},
    crs="EPSG:3006"
)

# ---------------------------
# Spatial join (points → grid)
# ---------------------------
joined = gpd.sjoin(gdf, grid, predicate="within")

# Sum population per grid cell
grid["Z"] = joined.groupby("index_right")["Z"].sum()
grid["Z"] = grid["Z"].fillna(0)

# Population density (people per km²)
grid["density"] = grid["Z"] / (50 * 50)

# ---------------------------
# Convert back to lat/lon
# ---------------------------
grid = grid.to_crs("EPSG:4326")
grid_plot = grid[grid["density"] > 0.05].copy()


# ---------------------------
# Plot with Plotly
# ---------------------------
fig = px.choropleth_map(
    grid_plot,
    geojson=grid_plot.geometry,
    locations=grid_plot.index,
    color="density",
    color_continuous_scale="YlOrRd",
    map_style="carto-positron",
    center={"lat": 62.0, "lon": 15.0},
    zoom=4.5,
    opacity=0.3,
    labels={"density": "People per km²"},
    title="Population Density in Sweden (50 km grid)"
)

df_airports = pd.read_csv('swe_airports.csv', sep=";")
icao = [str(code) for code in list(df_airports["ICAO"])]
iata = [str(code) for code in list(df_airports["IATA"])]
runways = [str(code) for code in list(df_airports["runway(s)"])]
labes = np.array([icao,iata,runways]).T
fig.add_trace(go.Scattermap(
    lon = df_airports['lon'],
    lat = df_airports['lat'],
    mode = 'markers',
    customdata= labes,
    hovertemplate="Coordinates: %{lon:.3f}°E %{lat:.3f}°N <br>"
                  "ICAO: %{customdata[0]}<br> IATA: %{customdata[1]}<br>"
                  "Runway(s): %{customdata[2]}  <extra></extra>",
    marker = dict(
        size = 10,
        color = 'rgb(255, 0, 0)',
    )))

# Initialize lists to store paths for plotting
latitudes = []
longitudes = []
airport_coords = df_airports[['lat', 'lon']].values
from geopy.distance import great_circle

# Generate great circle paths
start = airport_coords[88]
end = airport_coords[34]

# Interpolate great circle path points
lat, lon = great_circle_path(start[0], start[1], end[0], end[1], num_points=10)


# Flatten the lists of latitudes and longitudes
fig.add_trace(go.Scattermap(
    lat=lat,
    lon=lon,
    mode='lines',
))
fig.show()


