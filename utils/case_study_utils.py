import numpy as np
import pandas as pd
import logging

from utils.settings import settings

_logger = logging.getLogger(__name__)
_logger.setLevel(settings.logging_lvl)


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
        self.passengers = passengers
        self.curr_active = True if passengers else False

    def __str__(self):
        return f"Airport Name: {self.airport_name}\nICAO: {self.icao}\nIATA: {self.iata}\n" \
               f"Runways: {self.runways}\nLocation: ({self.lat}, {self.lon})\nAnnual Passengers: {self.passengers}"


def parse_runways(runways: str) -> list:
    """
    Helper method to parse runway information into a more usable format.

    Args:
    runways (str): A string for runways information structured as ((direction, length, runway surface), ...) .

    Returns:
        list: A list of information for each runway.
    """
    runways_list = []
    if runways:
        runway_details = runways.split('),')
        for runway in runway_details:
            runways_list.append(runway.replace('(', '').replace(')', ''))

    return runways_list


def get_airports(path: str) -> list:
    """
    Generates a list of Airports from a airports list .csv file.

    Args:
    path (str): Path to the airports list .csv file.

    Returns:
        list: A list of Airports
    """
    airports = []
    df_airports = pd.read_csv(path, sep=";").replace(np.nan, None)
    for cos in df_airports.iloc[1:].values.tolist():
        airports.append(Airport(*cos))

    _logger.info("Retrieved {} airports from airports list .csv file".format(len(airports)))
    return airports
