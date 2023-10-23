import ast
import copy
import logging
import os
from pathlib import Path

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
    assert os.path.exists(
        args.config_path
    ), f"Config file not found in {str(args.config_path)}"
    with open(args.config_path, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # Check for overridden parameters.
    #if args_override != []:
    #    overrides = create_dict_from_args(args_override)
    #    config, _ = merge_dicts(config, overrides)


    # Some other flags.
    if args.run_mode != None:
        config["task"]["run_mode"] = args.run_mode
    if args.seed != None:    
        config["task"]["seed"] = args.seed
    if args.gpu_id != None:
        config["task"]["gpu_id"] = args.gpu_id
    if args.src_dir != None:
        config["dataset"]["src"] = args.src_dir
    if args.batch_size != None:
        config["optim"]["batch_size"] = args.batch_size
    if args.pt_path != None:
        config["dataset"]["pt_path"] = args.pt_path
    if args.target_path != None:
        config["dataset"]["target_path"] = args.target_path
    if args.global_pool != None:
        config["model"]["pool"] = args.global_pool
    if args.lr != None:
        config["optim"]["lr"] = args.lr
    if args.clip != None:
        config["optim"]["clip_grad_norm"] = args.clip
    if args.num_epochs != None:
        config["optim"]["max_epochs"] = args.num_epochs
    if args.processed != None:
        processed = args.processed in ["True", "true", "1"]
        config["dataset"]["processed"] = processed
    if args.use_amp != None:
        use_amp = args.use_amp in ["True", "true", "1"]
        config["task"]["use_amp"] = use_amp
    if args.batch_tqdm != None:
        batch_tqdm = args.batch_tqdm in ["True", "true", "1"]
        config["optim"]["batch_tqdm"] = batch_tqdm
    if args.shut_otf != None:
        shut_otf = args.shut_otf in ["True", "true", "1"]
        if shut_otf:
            config["model"]["otf_edge"] = False
            config["model"]["gradient"] = False
            config["dataset"]["preprocess_params"]["preprocess_edges"] = True
            config["dataset"]["preprocess_params"]["preprocess_features"] = True
    #
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
