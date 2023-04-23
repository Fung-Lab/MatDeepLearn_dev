"""Launch jobs on local cluster or slurm in order to parallelize W&B sweep agents.
Works by generating job files that can be easily submitted to a cluster.
"""
import sys
import tempfile
from typing import Literal, Optional

# Define a training command template
training_command = lambda python_path, main_path, config_path, sweep_id: [
    python_path,
    main_path,
    "--config_path",
    config_path,
    "--use_wandb",
    "True",
    "--sweep_id",
    sweep_id,
    "--run_mode",
    "train",
]


def slurm_phoenix_entrypoint(
    config_path: str, main_path: str, sweep_id: str, slurm_job_config: dict
):
    # Load and create a SLURM batch command from the job config
    batch_commands = [
        f"#SBATCH -{key}{value}\n" for key, value in slurm_job_config["args"].items()
    ] + [f"#SBATCH {param}\n" for param in slurm_job_config["options"]]
    job_command = training_command(sys.executable, main_path, config_path, sweep_id)
    # create temporary job file
    temp = tempfile.NamedTemporaryFile(delete=False)
    with open(temp.name, "w") as tmp:
        tmp.writelines(batch_commands)
        tmp.write(" ".join(job_command) + "\n")
    temp.file.close()

    return temp.name


def cluster_entrypoint(config_path: str, main_path: str, sweep_id: str, index: int):
    # Create screen command and unique "job id"
    python_path = sys.executable
    screen_name = f"wandb_{sweep_id.split('/')[-1]}_{index}"
    screen_command = [
        "screen",
        "-S",
        screen_name,
        "-dm",
        python_path,
        main_path,
        "--config_path",
        config_path,
        "--use_wandb",
        "True",
        "--sweep_id",
        sweep_id,
        "--run_mode",
        "train",
    ]
    # execute the job file
    return screen_command


def start_sweep_tasks(
    system: Literal["slurm_phoenix", "local"],
    sweep_id: str,
    count: int,
    config_path: str,
    main_path: str,
    slurm_config: Optional[dict] = None,
) -> str:
    """Generate sweep tasks from sweep config"""
    temp = tempfile.NamedTemporaryFile(delete=False)
    with open(temp.name, "w") as tmp:
        tmp.write("#!/bin/sh\n")
        for i in range(count):
            if system == "phoenix_slurm":
                if slurm_config is None:
                    raise ValueError("SLURM config must be provided for SLURM system.")
                command = slurm_phoenix_entrypoint(
                    config_path, main_path, sweep_id, slurm_config
                )
            elif system == "local":
                # create temporary job file
                command = cluster_entrypoint(config_path, main_path, sweep_id, i)
            else:
                raise ValueError(
                    "System must be either 'phoenix_slurm', 'local', or 'perlmutter_slurm'"
                )
            tmp.write(" ".join(command) + "\n")
    temp.file.close()
    return temp.name
