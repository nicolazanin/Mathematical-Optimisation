import yaml
import os
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
    high_population_cells: list
    destination_cells: list


@dataclass
class AirportsConfig:
    num: int
    min_distance: int
    min_cost: int
    max_cost: int
    additional_airport_coords: list
    charging_bases_lim: int


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
    res: float
    max_total_time_travel: float
    min_ground_travel_time_to_destination_cell: float


@dataclass
class ModelConfig:
    epsilon: int
    mu_1: float
    mu_2: float
    lexicographic: bool
    mip_gap: float
    max_run_time: int


@dataclass
class HeuristicConfig:
    enable: bool
    initial_kernel_size: int
    buckets_size: int
    iterations: int
    max_no_improv_counter: int

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
    model_config: ModelConfig
    heuristic_config: HeuristicConfig
    random_seed: int
    logging_lvl: int
    print_logs: bool
    show_plot: bool
    simple_plot_enable: bool
    save_plot: bool

    @classmethod
    def from_yaml(cls, path: str) -> "Settings":
        """
        Load settings from a YAML file and return a Settings instance.

        Args:
            path (str): Path to the YAML configuration file.

        Returns:
            Settings: An instance of the Settings class populated with data from the file.
        """
        root_dir = os.path.abspath(os.path.dirname(__file__))
        config_path = os.path.join(root_dir, "../" + path)
        with open(config_path, "r") as f:
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

    log_file_name = "{}_{}.txt".format(
        log_prefix,
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    log_format = '%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s'
    date_format = '%H:%M:%S'

    root = logging.getLogger()
    root.setLevel(settings.logging_lvl)

    for handler in root.handlers[:]:
        handler.close()
        root.removeHandler(handler)

    console_formatter = logging.Formatter(log_format, date_format)
    console_formatter.converter = time.gmtime

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(settings.logging_lvl)
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)

    if print_file:
        file_formatter = logging.Formatter(log_format, date_format)
        file_formatter.converter = time.gmtime

        file_handler = logging.FileHandler(log_file_name, mode='a', delay=True)
        file_handler.setLevel(settings.logging_lvl)
        file_handler.setFormatter(file_formatter)
        root.addHandler(file_handler)

    root.info("Python version %s", sys.version)


settings = Settings.from_yaml("config.yml")
