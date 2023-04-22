"""Launch jobs on local cluster or slurm in order to parallelize W&B sweep agents"""
import subprocess
import tempfile
from typing import Literal, Optional


def slurm_perlmutter(sweep_id: str, slurm_job_config: dict, count: int = 1):
    raise NotImplementedError("Perlmutter capability WIP.")


def slurm_phoenix_entrypoint(sweep_id: str, slurm_job_config: dict, count: int = 1):
    # Load and create a SLURM batch command from the job config
    batch_commands = [
        f"#SBATCH -{key}{value}\n" for key, value in slurm_job_config["args"].items()
    ] + [f"#SBATCH {param}\n" for param in slurm_job_config["options"]]

    # create temporary job file
    temp = tempfile.NamedTemporaryFile(delete=True)

    with open(temp.name, "w") as tmp:
        tmp.writelines(batch_commands)
        tmp.write("conda activate matdeeplearn\n")
        tmp.write("wandb agent " + sweep_id + " --count " + str(count))
    temp.file.close()

    subprocess.run(["cat", temp.name], check=True)


def cluster_entrypoint(sweep_id: str, count: int = 1):
    # Create screen command and unique "job id"
    screen_command = "wandb agent " + sweep_id + " --count " + str(count)
    # launch sweep on the screen
    subprocess.run(
        ["screen", "-S", f"wandb_{sweep_id}", "-d", "-m", screen_command], check=True
    )


def start_sweep_tasks(
    system: Literal["slurm_phoenix", "local"],
    sweep_id: str,
    count: int,
    slurm_config: Optional[dict],
):
    """Generate sweep tasks from sweep config"""
    if system == "slurm_phoenix":
        if slurm_config is None:
            raise ValueError("SLURM config must be provided for SLURM system.")
        for _ in range(count):
            slurm_phoenix_entrypoint(sweep_id, slurm_config, 1)
    elif system == "local":
        cluster_entrypoint(sweep_id, count)
    elif system == "slurm_perlmutter":
        for _ in range(count):
            slurm_perlmutter(sweep_id, slurm_config, 1)
    else:
        raise ValueError(
            "System must be either 'slurm_phoenix', 'local', or 'slurm_perlmutter"
        )
