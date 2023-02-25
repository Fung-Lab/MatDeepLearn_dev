import contextlib
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
