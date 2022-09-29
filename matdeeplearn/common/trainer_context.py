# from matdeeplearn.common.utils import (
#     setup_logging,
#     setup_imports
# )

import copy
import time
from argparse import Namespace
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

@contextmanager
def new_trainer_context(*, config: Dict[str, Any], args: Namespace):
    # from matdeeplearn.common import distutils, gp_utils
    from matdeeplearn.common.registry import registry

    if TYPE_CHECKING:
        from matdeeplearn.tasks.task import BaseTask
        from matdeeplearn.trainers import BaseTrainer

    @dataclass
    class _TrainingContext:
        config: Dict[str, Any]
        task: "BaseTask"
        trainer: "BaseTrainer"

    # setup_logging()
    config = copy.deepcopy(config)

    # TODO: set up for distributed system
    # if args.distributed:
    #     distutils.setup(config)
    #     if config["gp_gpus"] is not None:
    #         gp_utils.setup_gp(config)
    try:
        # setup_imports(config)
        trainer_cls = registry.get_trainer_class(
            config.get("trainer", "property")
        )
        assert trainer_cls is not None, "Trainer not found"

        #TODO: set up trainer to include different attributes from matedeeplearn
        trainer = trainer_cls(
            task=config["task"],
            model=config["model"],
            dataset=config["dataset"],
            optimizer=config["optim"],
            rank=config["rank"],
            verbosity=config["verboxity"]
        )

        task_cls = registry.get_task_class(config["run_mode"])
        assert task_cls is not None, "Task not found"
        task = task_cls(config)
        start_time = time.time()
        ctx = _TrainingContext(config=config, task=task, trainer=trainer)
        yield ctx

        #TODO: add in distributed system functionality
        # distutils.synchronize()
        # if distutils.is_master():
        #     logging.info(f"Total time taken: {time.time() - start_time}")
    finally:
        pass
    #     if args.distributed:
    #         distutils.cleanup()
