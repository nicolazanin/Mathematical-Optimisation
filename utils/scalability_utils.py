import pandas as pd
from utils.settings import Settings
import yaml


def apply_preset(settings_to_mod: Settings, presets_path: str, preset_name: str) -> None:
    """
    Loads presets from a YAML and modifies the Settings object in-place.
    """
    with open(presets_path, "r") as f:
        all_presets = yaml.safe_load(f)

    preset = all_presets[preset_name]
    settings_to_mod.model_config.mu_1 = preset["mu_1"]
    settings_to_mod.model_config.mu_2 = preset["mu_2"]
    settings_to_mod.model_config.lexicographic = preset["lexicographic"]
    settings_to_mod.model_config.mip_gap = preset["mip_gap"]
    settings_to_mod.heuristic_config.enable = preset["enable"]
    settings_to_mod.heuristic_config.iterations = preset["iterations"]


def init_results_dict():
    results = {
        "test": [], "N": [], "K": [], "tau": [],
        "b&c": {"rows": [], "cols": [], "obj_1": [], "obj_2": [], "bound": [], "t": []},
        "kn_1": {"obj_1": [], "obj_2": [], "bound": [], "t": []},
        "kn_3": {"obj_1": [], "obj_2": [], "bound": [], "t": []}
    }
    return results


def flatten_results(results):
    # Flatten nested dictionaries into separate columns
    flattened = {key: [] for key in results if not isinstance(results[key], dict)}

    for subkey in ["b&c", "kn_1", "kn_3"]:
        if subkey in results:
            for inner_key, values in results[subkey].items():
                new_key = f"{subkey}_{inner_key}"
                flattened[new_key] = values
    for key in results:
        if not isinstance(results[key], dict):
            flattened[key] = results[key]

    return flattened


def print_report(results, name: str) -> None:
    flattened_results = flatten_results(results)
    df = pd.DataFrame(flattened_results)
    file_name = f"{name}.csv"
    df.to_csv(file_name, index=False)
