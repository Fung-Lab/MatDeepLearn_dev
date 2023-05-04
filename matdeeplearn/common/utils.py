import contextlib
import logging
from itertools import combinations, product

import pandas
import torch
from torch.profiler import ProfilerActivity, profile

import numpy as np


def argmax(arr: list[dict], key: str) -> int:
    """List of Dict argmax utility function

    Args:
        arr (list[dict]): _description_
        key (str): _description_

    Returns:
        _type_: _description_
    """
    return max(enumerate(arr), key=lambda x: x.get(key))[0]


def subsets(arr: set) -> list:
    subsets = []
    [subsets.extend(list(combinations(arr, n))) for n in range(len(arr) + 1)]
    return subsets[1:]


def generate_mp_combos(mp_attrs: dict, num_layers) -> list:
    return [
        list([list(y) for y in x])
        for x in product(subsets(mp_attrs), repeat=num_layers)
    ]


def get_mae_from_preds():
    df = pandas.read_csv(os.path.join(sys.argv[1], "test_predictions.csv"))
    # pred_comp = df.filter(["target", "prediction"])
    return np.abs(df["target"] - df["prediction"]).mean()


@contextlib.contextmanager
def prof_ctx(profile_key="CPU"):
    """Primitive debug tool which allows profiling of PyTorch code"""
    with profile(
        activities=[
            ProfilerActivity.CUDA if profile_key == "CUDA" else ProfilerActivity.CPU
        ],
        record_shapes=True,
        profile_memory=True,
    ) as prof:

        yield

    logging.info(prof.key_averages().table(row_limit=10))


def min_alloc_gpu(device: str = None):
    """Returns the GPU with least allocated memory for training

    Returns:
        torch.device: GPU with least allocated memory, MPS backend if available, CPU otherwise
    """
    # MPS and CUDA support
    if device and device.startswith("cuda") and torch.cuda.is_available():
        # check device ordinal validity
        if int(device[-1]) >= torch.cuda.device_count():
            raise ValueError("Invalid CUDA device ordinal, fix device choice in config")
        logging.debug(f"Using CUDA device {device}")
        return torch.device(device)
    else:
        if torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                raise AssertionError(
                    "MPS not available because the current PyTorch install was not built with MPS enabled."
                )
            logging.debug("Using MPS backend")
            return torch.device("mps")
        elif torch.cuda.is_available():
            # get the GPU with least allocated memory
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            device = torch.device("cuda:0")
            for i in range(1, torch.cuda.device_count()):
                mem = torch.cuda.memory_allocated(i)
                if mem < gpu_mem:
                    gpu_mem = mem
                    device = torch.device(f"cuda:{i}")
            return device
        else:
            logging.warning("GPU or MPS is not available, defaulting to train on CPU")
            return torch.device("cpu")


class DictTools:
    """Useful static dict tools for working with nested dicts"""

    @staticmethod
    def _get_nested(data, *args):
        if args and data:
            element = args[0]
            if element and element in data:
                value = data.get(element)
                return (
                    value if len(args) == 1 else DictTools._get_nested(value, *args[1:])
                )

    @staticmethod
    def _flatten(my_dict):
        result = {}
        for key, value in my_dict.items():
            if isinstance(value, dict):
                result.update(DictTools._flatten(value))
            else:
                result[key] = value
        return result

    @staticmethod
    def _mod_recurse(obj: dict, key: str, item):
        if key in obj:
            obj[key] = item
            return
        for _, v in obj.items():
            if isinstance(v, dict):
                DictTools._mod_recurse(v, key, item)
            elif isinstance(v, list):
                for e in v:
                    if isinstance(e, dict):
                        DictTools._mod_recurse(e, key, item)

    @staticmethod
    def _convert_to_list(obj: dict):
        """Pointer reinforcement approach"""
        if not isinstance(obj, dict):
            return obj
        for k, v in obj.items():
            if isinstance(v, list):
                obj[k] = dict()
                for i, item in enumerate(v):
                    obj[k][str(i)] = DictTools._convert_to_list(item)
                return obj
            else:
                obj[k] = DictTools._convert_to_list(v)
                return obj
