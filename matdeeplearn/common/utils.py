import contextlib
import torch
import logging
from torch.profiler import ProfilerActivity, profile


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
        torch.device: GPU with least allocated memory
    """

    if device:
        return torch.device(device)

    if not torch.cuda.is_available():
        logging.warning("GPU is not available, proceeding to train on CPU")
        return torch.device("cpu")

    # get the GPU with least allocated memory
    gpu_mem = torch.cuda.get_device_properties(0).total_memory
    device = torch.device("cuda:0")
    for i in range(1, torch.cuda.device_count()):
        mem = torch.cuda.memory_allocated(i)
        if mem < gpu_mem:
            gpu_mem = mem
            device = torch.device(f"cuda:{i}")

    return device


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
            print(f"Modifying {key} in {obj} to {item}")
            obj[key] = item
            return
        for _, v in obj.items():
            if isinstance(v, dict):
                DictTools._mod_recurse(v, key, item)

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
