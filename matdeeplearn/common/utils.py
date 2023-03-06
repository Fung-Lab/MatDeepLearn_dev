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


def min_alloc_gpu():
    """Returns the GPU with least allocated memory for training

    Returns:
        torch.device: GPU with least allocated memory
    """

    if not torch.cuda.is_available():
        logging.warning("GPU is not available, proceeding to train on CPU")
        return torch.device("cpu")

    # get the GPU with least allocated memory
    gpu_mem = torch.cuda.get_device_properties(0).total_memory
    device = torch.device("cuda:0")
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.memory_allocated(i)
        if mem < gpu_mem:
            gpu_mem = mem
            device = torch.device(f"cuda:{i}")

    return device
