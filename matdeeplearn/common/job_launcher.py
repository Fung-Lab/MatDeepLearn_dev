"""Launch jobs on local cluster or slurm in order to parallelize W&B sweep agents"""
import subprocess
import os
import sys
import tempfile
from typing import Literal, Optional


def slurm_phoenix_entrypoint(
    config_path: str, main_path: str, sweep_id: str, slurm_job_config: dict
):
    # Load and create a SLURM batch command from the job config
    batch_commands = [
        f"#SBATCH -{key}{value}\n" for key, value in slurm_job_config["args"].items()
    ] + [f"#SBATCH {param}\n" for param in slurm_job_config["options"]]

    # create temporary job file
    temp = tempfile.NamedTemporaryFile(delete=True)
    wandb_path = subprocess.check_output(["which", "wandb"]).decode("utf-8").strip()

    with open(temp.name, "w") as tmp:
        tmp.writelines(batch_commands)
        tmp.write("conda activate matdeeplearn\n")
        tmp.write(f"{wandb_path} agent {sweep_id} --count {1}")
    temp.file.close()

    return subprocess.run(["sbatch", temp.name], check=True, capture_output=True)


def cluster_entrypoint(config_path: str, main_path: str, sweep_id: str):
    # Create screen command and unique "job id"
    python_path = sys.executable
    screen_command = [
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
    # launch sweep on the screen
    screen_name = f"wandb_{sweep_id.split('/')[-1]}"
    subprocess.run(
        ["screen", "-S", screen_name, "-d", "-m"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return subprocess.run(
        [
            "screen",
            "-r",
            screen_name,
            "-X",
            "stuff",
            " ".join(screen_command),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def start_sweep_tasks(
    system: Literal["slurm_phoenix", "local"],
    sweep_id: str,
    count: int,
    config_path: str,
    main_path: str,
    slurm_config: Optional[dict] = None,
):
    """Generate sweep tasks from sweep config"""
    if system == "slurm_phoenix":
        if slurm_config is None:
            raise ValueError("SLURM config must be provided for SLURM system.")
        for _ in range(count):
            slurm_phoenix_entrypoint(config_path, main_path, sweep_id, slurm_config)
    elif system == "local":
        for _ in range(count):
            out = cluster_entrypoint(config_path, main_path, sweep_id)
            print(" ".join(out.args))
    else:
        raise ValueError(
            "System must be either 'slurm_phoenix', 'local', or 'slurm_perlmutter"
        )
