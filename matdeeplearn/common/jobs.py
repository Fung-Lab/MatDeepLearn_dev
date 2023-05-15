"""
Streamlined way to abstract jobs and runs.
"""

import os
import tempfile
from abc import ABC, abstractmethod
from typing import Optional

import yaml

from matdeeplearn.common.registry import registry
from matdeeplearn.tasks.task import BaseTask

CONFIG_PATH = os.path.join(os.path.join(os.getcwd(), os.pardir), "configs")


@registry.register_job("base")
class Job(ABC):
    def __init__(self, name: str, type: str, slurm: bool) -> None:
        self.name = name
        self.slurm = slurm
        with open(os.path.join(CONFIG_PATH, "jobs", f"{type}.yml"), "r") as f:
            data = yaml.full_load(f)
        self.job_config = data

    @abstractmethod
    def entrypoint(
        self,
        config_path: str,
        use_wandb: bool,
        task: str,
        sweep_id: Optional[tuple[int, str]] = None,
    ):
        pass


@registry.register_job("slurm")
@registry.register_job("phoenix_slurm")
@registry.register_job("perlmutter")
class SlurmJob(Job):
    def __init__(self, name: str, type: str) -> None:
        super().__init__(name, type, slurm=True)

    def entrypoint(
        self,
        config_path: str,
        use_wandb: bool,
        task: str,
        sweep_id: Optional[tuple[int, str]] = None,
    ) -> str:
        # Load and create a SLURM batch command from the job config
        batch_commands = (
            [f"#SBATCH -J{self.name}\n", f"#SBATCH -o{self.name}-%j.out\n"]
            + [
                f"#SBATCH -{key}{value}\n"
                for key, value in self.job_config["args"].items()
            ]
            + [f"#SBATCH {param}\n" for param in self.job_config["options"]]
        )
        job_command = training_command(
            config_path, task, use_wandb, sweep_id=sweep_id[1] if sweep_id else None
        )

        # create temporary job script file
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".sh")
        with open(temp.name, "w") as tmp:
            tmp.write("#!/bin/bash\n")
            tmp.writelines(batch_commands)
            tmp.write(job_command + "\n")
        temp.file.close()

        return f"sbatch {temp.name}"


@registry.register_job("local")
class LocalJob(Job):
    def __init__(self, name: str, type: str) -> None:
        super().__init__(name, type, slurm=False)

    def entrypoint(
        self,
        config_path: str,
        use_wandb: bool,
        task: str,
        sweep_id: Optional[tuple[int, str]] = None,
    ) -> str:
        screen_name = (
            f"wandb_{sweep_id[1].split('/')[-1]}_{sweep_id[0]}" if sweep_id else ""
        )
        # use screen for local jobs
        screen_command = ["screen", "-S", screen_name, "-dm"]
        job_command = training_command(
            config_path, task, use_wandb, sweep_id=sweep_id[1] if sweep_id else None
        )
        return f"{' '.join(screen_command)} {job_command}"


def training_command(
    config_path, task: str, use_wandb: bool, sweep_id: Optional[str] = None
):
    # get conda env path
    conda_env_path = os.environ.get("CONDA_PREFIX")
    # setup command
    if registry.get_task_class(task) == BaseTask:
        raise ValueError(f"Task {task} not found in registry.")
    command = [
        os.path.join(conda_env_path, "bin/python"),
        os.path.join(__file__, "main.py"),
        f"--config_path={config_path}",
        f"--use_wandb={use_wandb}",
        f"--run_mode={task}",
        f"--sweep_id={sweep_id}",
    ]

    return " ".join(command)


def start_sweep_tasks(
    system: str,
    sweep_id: str,
    count: int,
    config_path: str,
    identifier: str,
    use_wandb: bool,
    task: str,
) -> str:
    """Generate sweep tasks job scripts from sweep config

    Args:
        system (str): _description_
        sweep_id (str): _description_
        count (int): _description_
        config_path (str): _description_
        identifier (str): _description_

    Returns:
        str: _description_
    """

    assert (
        system in registry.mapping["jobs"].keys()
    ), f"System {system} not found in registry."

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".sh")
    with open(temp.name, "w") as tmp:
        tmp.write("#!/bin/sh\n")
        for i in range(count):
            job_cls = registry.get_job_class(system)
            job = job_cls(f"{identifier}_{i}" if i > 0 else identifier)
            tmp.write(
                f"sh {job.entrypoint(config_path, use_wandb, task, sweep_id)}" + "\n"
            )
    temp.file.close()
    return temp.name
