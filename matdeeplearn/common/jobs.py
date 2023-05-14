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


@registry.register_job("base")
class Job(ABC):
    def __init__(self, name: str, slurm: bool) -> None:
        self.name = name
        self.slurm = slurm
        with open(CONFIG_PATH, "jobs", f"{name}.yml", "r") as f:
            self.slurm_job_config = yaml.full_load(f)

    @abstractmethod
    def entrypoint(
        self,
        config_path: str,
        use_wandb: bool,
        task: str,
        sweep_id: Optional[tuple[int, str]],
    ):
        pass


@registry.register_job("slurm")
class SlurmJob(Job):
    def __init__(self, name: str) -> None:
        super().__init__(name, slurm=True)

    def entrypoint(
        self,
        config_path: str,
        use_wandb: bool,
        task: str,
        sweep_id: Optional[tuple[int, str]],
    ) -> str:
        # Load and create a SLURM batch command from the job config
        batch_commands = (
            [f"#SBATCH -J{self.name}"]
            + [
                f"#SBATCH -{key}{value}\n"
                for key, value in self.slurm_job_config["args"].items()
            ]
            + [f"#SBATCH {param}\n" for param in self.slurm_job_config["options"]]
        )
        job_command = training_command(config_path, task, use_wandb, sweep_id)

        # create temporary job script file
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".sh")
        with open(temp.name, "w") as tmp:
            tmp.writelines(batch_commands)
            tmp.write(" ".join(job_command) + "\n")
        temp.file.close()

        return f"sbatch {temp.name}"


@registry.register_job("local")
class LocalJob(Job):
    def __init__(self, name: str) -> None:
        super().__init__(name, slurm=False)

    def entrypoint(
        self,
        config_path: str,
        use_wandb: bool,
        task: str,
        sweep_id: Optional[tuple[int, str]],
    ) -> str:
        screen_name = (
            f"wandb_{sweep_id[1].split('/')[-1]}_{sweep_id[0]}" if sweep_id else ""
        )
        # use screen for local jobs
        screen_command = ["screen", "-S", screen_name, "-dm"]
        job_command = training_command(config_path, task, use_wandb, sweep_id)
        return " ".join(screen_command + job_command)


def training_command(config_path, task: str, use_wandb: bool, sweep_id: Optional[str]):
    # get conda env path
    conda_env_path = os.environ.get("CONDA_PREFIX")
    # setup command
    if registry.get_task_class(task) == BaseTask:
        raise ValueError(f"Task {task} not found in registry.")
    command = [
        os.path.join(conda_env_path, "bin/python"),
        os.path.join(__file__, "main.py"),
        "--config_path",
        config_path,
        "--use_wandb",
        use_wandb,
        "--run_mode",
        task,
        "--sweep_id",
        sweep_id,
    ]

    return command


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
