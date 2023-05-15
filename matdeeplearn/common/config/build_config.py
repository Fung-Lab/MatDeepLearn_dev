import ast
import copy
import os
import logging

from matdeeplearn.common.utils import DictTools
from matdeeplearn.common.jobs import CONFIG_PATH

import yaml


def merge_dicts(dict1: dict, dict2: dict):
    """Recursively merge two dictionaries.
    Values in dict2 override values in dict1. If dict1 and dict2 contain a dictionary as a
    value, this will call itself recursively to merge these dictionaries.
    This does not modify the input dictionaries (creates an internal copy).
    Additionally returns a list of detected duplicates.
    Adapted from https://github.com/TUM-DAML/seml/blob/master/seml/utils.py

    Parameters
    ----------
    dict1: dict
        First dict.
    dict2: dict
        Second dict. Values in dict2 will override values from dict1 in case they share the same key.

    Returns
    -------
    return_dict: dict
        Merged dictionaries.
    """
    if not isinstance(dict1, dict):
        raise ValueError(f"Expecting dict1 to be dict, found {type(dict1)}.")
    if not isinstance(dict2, dict):
        raise ValueError(f"Expecting dict2 to be dict, found {type(dict2)}.")

    return_dict = copy.deepcopy(dict1)
    duplicates = []

    for k, v in dict2.items():
        if k not in dict1:
            return_dict[k] = v
        else:
            if isinstance(v, dict) and isinstance(dict1[k], dict):
                return_dict[k], duplicates_k = merge_dicts(dict1[k], dict2[k])
                duplicates += [f"{k}.{dup}" for dup in duplicates_k]
            else:
                return_dict[k] = dict2[k]
                duplicates.append(k)

    return return_dict, duplicates


def dict_set_recursively(dictionary, key_sequence, val):
    top_key = key_sequence.pop(0)
    if len(key_sequence) == 0:
        dictionary[top_key] = val
    else:
        if top_key not in dictionary:
            dictionary[top_key] = {}
        dict_set_recursively(dictionary[top_key], key_sequence, val)


def parse_value(value):
    """
    Parse string as Python literal if possible and fallback to string.
    """
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # Use as string if nothing else worked
        return value


def create_dict_from_args(args: list, sep: str = "."):
    """
    Create a (nested) dictionary from console arguments.
    Keys in different dictionary levels are separated by sep.
    """
    return_dict = {}
    for arg in args:
        arg = arg.strip("--")
        keys_concat, val = arg.split("=")
        val = parse_value(val)
        key_sequence = keys_concat.split(sep)
        dict_set_recursively(return_dict, key_sequence, val)
    return return_dict


def build_config(args, args_override):
    # Open provided config file
    if os.path.exists(args.config_path):
        config_path = args.config_path
        logging.info(f"Using config file: {config_path}")
    else:
        # using a config file template
        templates = {
            file.strip(".yml"): os.path.join(CONFIG_PATH, "config_templates", str(file))
            for file in os.listdir(os.path.join(CONFIG_PATH, "config_templates"))
            if file.endswith(".yml")
        }
        config_path = templates[str(args.config_path)]
        logging.info(f"Using config file template: {config_path}")

    with open(config_path, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # Check for overridden parameters.
    if args_override != []:
        overrides = create_dict_from_args(args_override)
        logging.debug(f"Overridden parameters: {list(overrides.keys())}")
        for key, item in overrides.items():
            DictTools._mod_recurse(config, key, item)

    # Some other flags.
    config["run_mode"] = args.run_mode
    config["seed"] = args.seed

    # Submit
    config["submit"] = args.submit
    # config["summit"] = args.summit
    # Distributed
    # TODO: add distributed flags

    # if run_mode != "Hyperparameter":
    #
    #     process_start_time = time.time()
    #
    #     dataset = preprocessor.get_dataset(
    #         config["Processing"]["data_path"],
    #         config["Training"]["target_index"],
    #         config["Job"]["reprocess"],
    #         config["Processing"],
    #     )

    return config
