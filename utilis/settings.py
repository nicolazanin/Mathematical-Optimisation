import yaml
from dataclasses import dataclass
from dacite import from_dict
import logging
import time
import sys
import datetime

_is_setup_done = False


@dataclass
class PopulationConfig:
    cells_x: int
    cells_y: int
    cell_area: int
    min_density: int
    max_density: int
    destination_cells: list


@dataclass
class AirportsConfig:
    num: int
    min_distance: int


@dataclass
class AircraftConfig:
    tau: int
    cruise_speed: int


@dataclass
class GroundAccessConfig:
    max_time: int
    avg_speed: int


@dataclass
class PathsConfig:
    max_edges: int
    routing_factor_thr: float


@dataclass
class Settings:
    """
    Container for application configuration loaded from a YAML file.

    Attributes:
        population_config (PopulationConfig): Configuration for population settings.
        airports_config (AirportsConfig): Configuration for airport settings.
        aircraft_config (AircraftConfig): Configuration for aircraft settings.
        ground_access_config (GroundAccessConfig): Configuration for ground access settings.
        paths_config (PathsConfig): Configuration for path-related settings.
        logging_lvl (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        print_logs (bool): If True, print logs to the console or file.

    Methods:
        from_yaml(path: str) -> Settings:
            Class method to create a Settings instance from a YAML file.
    """
    population_config: PopulationConfig
    airports_config: AirportsConfig
    aircraft_config: AircraftConfig
    ground_access_config: GroundAccessConfig
    paths_config: PathsConfig
    logging_lvl: int
    print_logs: bool

    @classmethod
    def from_yaml(cls, path: str) -> "Settings":
        """
        Load settings from a YAML file and return a Settings instance.

        Args:
            path (str): Path to the YAML configuration file.

        Returns:
            Settings: An instance of the Settings class populated with data from the file.
        """
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        return from_dict(data_class=cls, data=raw)


def setup_logging(log_prefix: str, print_file: bool = True) -> None:
    """
    Set up the root logger instance and connect all the handlers.

    Args:
        log_prefix (str): Prefix for the log file name.
        print_file (bool): If True, enable printing log output to file, if False, disable file logging.

    Returns:
        None
    """

    global _is_setup_done

    if _is_setup_done:
        pass
    else:
        _log_file_name = "{}-{}_log_file.txt".format(log_prefix, datetime.datetime.utcnow().
                                                     strftime("%Y-%m-%d %H:%M:%S").replace(":", "-").replace(" ", "-"))
        _log_format = '%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s'
        _console_date_format = '%Y-%m-%d %H:%M:%S'
        _file_date_format = _console_date_format

        console_formatter = logging.Formatter(_log_format, _console_date_format)
        console_formatter.converter = time.gmtime
        file_formatter = logging.Formatter(_log_format, _file_date_format)
        file_formatter.converter = time.gmtime

        root = logging.getLogger()
        root.setLevel(logging.INFO)

        if len(root.handlers) != 0:
            console_handler = root.handlers[0]
        else:
            console_handler = logging.StreamHandler(sys.stderr)

        file_handler = logging.FileHandler(_log_file_name, mode='a', delay=True)

        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(console_formatter)

        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)

        root.addHandler(console_handler)

        if print_file:
            root.addHandler(file_handler)

        root.info("Python version %s" % sys.version)

        _is_setup_done = True


settings = Settings.from_yaml("config.yml")
