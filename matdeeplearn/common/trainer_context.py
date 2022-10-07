# from matdeeplearn.common.utils import (
#     setup_logging,
#     setup_imports
# )

import copy
import time
import importlib
from argparse import Namespace
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
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
        setup_imports(config)
        trainer_cls = registry.get_trainer_class(
            config.get("trainer", "property")
        )
        assert trainer_cls is not None, "Trainer not found"

        #TODO: set up trainer to include different attributes from matedeeplearn
        trainer = trainer_cls.from_config(
            task=config["task"],
            model=config["model"],
            dataset_config=config["dataset"],
            # optimizer=config["optim"],
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

def _import_local_file(path: Path, *, project_root: Path):
    """
    Imports a Python file as a module

    :param path: The path to the file to import
    :type path: Path
    :param project_root: The root directory of the project (i.e., the "ocp" folder)
    :type project_root: Path
    """

    path = path.resolve()
    project_root = project_root.resolve()

    module_name = ".".join(
        path.absolute()
        .relative_to(project_root.absolute())
        .with_suffix("")
        .parts
    )
    # logging.debug(f"Resolved module name of {path} to {module_name}")
    importlib.import_module(module_name)


def setup_experimental_imports(project_root: Path):
    experimental_folder = (project_root / "experimental").resolve()
    if not experimental_folder.exists() or not experimental_folder.is_dir():
        return

    experimental_files = [
        f.resolve().absolute() for f in experimental_folder.rglob("*.py")
    ]
    # Ignore certain directories within experimental
    ignore_file = experimental_folder / ".ignore"
    if ignore_file.exists():
        with open(ignore_file, "r") as f:
            for line in f.read().splitlines():
                for ignored_file in (experimental_folder / line).rglob("*.py"):
                    experimental_files.remove(
                        ignored_file.resolve().absolute()
                    )

    for f in experimental_files:
        _import_local_file(f, project_root=project_root)


def _get_project_root():
    """
    Gets the root folder of the project (the "ocp" folder)
    :return: The absolute path to the project root.
    """
    from matdeeplearn.common.registry import registry

    # Automatically load all of the modules, so that
    # they register with registry
    root_folder = registry.get("ocpmodels_root", no_warning=True)

    if root_folder is not None:
        assert isinstance(root_folder, str), "ocpmodels_root must be a string"
        root_folder = Path(root_folder).resolve().absolute()
        assert root_folder.exists(), f"{root_folder} does not exist"
        assert root_folder.is_dir(), f"{root_folder} is not a directory"
    else:
        root_folder = Path(__file__).resolve().absolute().parent.parent

    # root_folder is the "ocpmodes" folder, so we need to go up one more level
    return root_folder.parent


# Copied from https://github.com/facebookresearch/mmf/blob/master/mmf/utils/env.py#L89.
def setup_imports(config: Optional[dict] = None):
    from matdeeplearn.common.registry import registry

    skip_experimental_imports = (config or {}).get(
        "skip_experimental_imports", None
    )

    # First, check if imports are already setup
    has_already_setup = registry.get("imports_setup", no_warning=True)
    if has_already_setup:
        return

    try:
        project_root = _get_project_root()
        # logging.info(f"Project root: {project_root}")
        # importlib.import_module("ocpmodels.common.logger")

        import_keys = ["trainers", "datasets", "models", "tasks"]
        for key in import_keys:
            for f in (project_root / "matdeeplearn" / key).rglob("*.py"):
                _import_local_file(f, project_root=project_root)

        if not skip_experimental_imports:
            setup_experimental_imports(project_root)
    finally:
        registry.register("imports_setup", True)
