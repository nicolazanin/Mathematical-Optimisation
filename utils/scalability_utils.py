import pandas as pd
from utils.settings import Settings
import yaml


def apply_preset(settings_to_mod: Settings, presets_path: str, test_name: str) -> None:
    """
    Loads specific configuration presets from a YAML file and updates the Settings object to perform scalability tests.

    Args:
        settings_to_mod (Settings): The settings object to be modified.
        presets_path (str): The file path to the YAML containing the presets for scalability tests.
        test_name (str): The specific test name key in the YAML file to extract settings from.

    Returns:
        None
    """
    with open(presets_path, "r") as f:
        all_presets = yaml.safe_load(f)

    preset = all_presets[test_name]
    settings_to_mod.model_config.mu_1 = preset["mu_1"]
    settings_to_mod.model_config.mu_2 = preset["mu_2"]
    settings_to_mod.model_config.lexicographic = preset["lexicographic"]
    settings_to_mod.model_config.mip_gap = preset["mip_gap"]
    settings_to_mod.heuristic_config.enable = preset["enable"]
    settings_to_mod.heuristic_config.iterations = preset["iterations"]
    settings_to_mod.heuristic_config.max_no_improv_counter = preset["max_no_improv_counter"]


def init_results_dict() -> dict:
    """
    Initializes a structured dictionary to store experimental results.

    Returns:
        dict: A dictionary structure ready to be populated with metric data.
    """
    results = {
        "test": [], "N": [], "K": [], "tau": [],
        "b&c": {"rows": [], "cols": [], "obj_1": [], "obj_2": [], "bound": [], "t": []},
        "kn_1": {"obj_1": [], "obj_2": [], "bound": [], "t": []},
        "kn_3": {"obj_1": [], "obj_2": [], "bound": [], "t": []},
        "kn_3_l": {"obj_1": [], "obj_2": [], "bound": [], "t": []}
    }
    return results


def flatten_results(results: dict) -> dict:
    """
    Flattens a nested results dictionary into a single-level format.

    Args:
        results (dict): The nested dictionary containing experimental metrics.

    Returns:
        dict: A flattened dictionary where each key corresponds to a unique data column.
    """
    # Flatten nested dictionaries into separate columns
    flattened = {key: [] for key in results if not isinstance(results[key], dict)}

    for subkey in ["b&c", "kn_1", "kn_3", "kn_3_l"]:
        if subkey in results:
            for inner_key, values in results[subkey].items():
                new_key = f"{subkey}_{inner_key}"
                flattened[new_key] = values
    for key in results:
        if not isinstance(results[key], dict):
            flattened[key] = results[key]

    return flattened


def print_report(results: dict, name: str) -> None:
    """
    Converts the results dictionary to a Pandas DataFrame and exports it to CSV.

    Args:
        results (dict): The (potentially nested) dictionary of results.
        name (str): The base filename for the output (the .csv extension is added).

    Returns:
        None
    """
    flattened_results = flatten_results(results)
    df = pd.DataFrame(flattened_results)
    file_name = f"{name}.csv"
    df.to_csv(file_name, index=False)
