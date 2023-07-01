import hashlib
import logging
import os
import pprint
import tempfile
from datetime import datetime

import torch
import wandb
import yaml
from torch import distributed as dist

from matdeeplearn.common.config.build_config import build_config
from matdeeplearn.common.config.flags import flags
from matdeeplearn.common.jobs import CONFIG_PATH, start_sweep_tasks
from matdeeplearn.common.registry import registry
from matdeeplearn.common.trainer_context import new_trainer_context
from matdeeplearn.common.utils import DictTools
from matdeeplearn.preprocessor.processor import process_data

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.autograd.set_detect_anomaly(True)

torch.set_printoptions(profile="full")


class Runner:
    def __init__(self):
        self.config = None

    def __call__(self, config, args):
        with new_trainer_context(args=args, config=config) as ctx:
            self.config = ctx.config
            self.task = ctx.task
            self.trainer = ctx.trainer

            self.task.setup(self.trainer)

            # Print settings for job
            logging.debug("Settings: ")
            logging.debug(pprint.pformat(self.config))

            self.task.run()

    def checkpoint(self, *args, **kwargs):
        # new_runner = Runner()
        self.trainer.save(checkpoint_file="checkpoint.pt", training_state=True)
        self.config["checkpoint"] = self.task.chkpt_path
        self.config["timestamp_id"] = self.trainer.timestamp_id
        if self.trainer.logger is not None:
            self.trainer.logger.mark_preempting()


def wandb_setup(config):
    metadata = config["task"]["wandb"].get("metadata", {})
    _wandb_config = {
        "metadata": metadata,
    }
    # update config with chosen parameters from config (to avoid clutter)
    track_params = [
        p.split(".") for p in config["task"]["wandb"].get("track_params", {})
    ]
    track_params = {p[-1]: DictTools._get_nested(config, *p) for p in track_params}
    # transform hyperparams
    transforms = config["dataset"]["transforms"]
    transforms = {
        key: val
        for d in transforms
        for key, val in DictTools._flatten(d).items()
        if key != "name" and key != "otf"
    }

    _wandb_config.update(transforms)
    _wandb_config.update(track_params)

    # Compute a permutation-invariant hash of the preprocessing parameters for each run
    preprocess_params = config["dataset"]["preprocess_params"]
    transforms = config["dataset"]["transforms"]

    sorted_transforms = sorted(
        [x for transform in transforms for x in DictTools._flatten(transform).items()],
        key=lambda x: x[0],
    )
    sorted_preprocess_params = sorted(
        DictTools._flatten(preprocess_params).items(), key=lambda x: x[0]
    )
    sorted_model_hyperparams = sorted(
        DictTools._flatten(config["model"]["hyperparams"]).items(), key=lambda x: x[0]
    )

    hash_str = hashlib.md5(
        (
            str(sorted_transforms)
            + str(sorted_preprocess_params)
            + str(sorted_model_hyperparams)
        ).encode("utf-8")
    )
    _wandb_config["meta_hash"] = hash_str.hexdigest()

    if config["task"]["run_id"] != "" and not config["model"]["load_model"]:
        raise ValueError("Must load from checkpoint if also resuming wandb run.")

    wandb_id = config["task"]["run_id"] if config["task"]["run_id"] != "" else None
    wandb.init(
        settings=wandb.Settings(start_method="fork"),
        project=config["task"]["wandb"].get("wandb_project", "matdeeplearn"),
        entity=config["task"]["wandb"].get("wandb_entity", "fung-lab"),
        name=f"{config['task']['identifier']}-{timestamp}",
        notes=config["task"]["wandb"].get("notes", None),
        tags=config["task"]["wandb"].get("tags", None),
        config=_wandb_config,
        id=wandb_id,
        resume="must" if config["task"]["run_id"] != "" else None,
    )
    # create wandb artifacts
    if not config["task"]["wandb"]["sweep"]["do_sweep"]:
        wandb_artifacts = config["task"]["wandb"].get("log_artifacts", [])
        for _, artifact in enumerate(wandb_artifacts):
            if not os.path.exists(str(artifact)):
                raise ValueError(
                    f"Artifact {str(artifact)} does not exist. Please check the path."
                )
            else:
                wandb.save(artifact)
    else:
        wandb.run.name = f"{wandb.run.name}-{wandb.run.id}"


def main():
    """Entrypoint for MatDeepLearn inference tasks"""
    if config["task"]["wandb"]["use_wandb"]:
        wandb_setup(config)
    if not config["dataset"]["processed"]:
        process_data(config["dataset"], config["task"]["wandb"])
    Runner()(config, args)


if __name__ == "__main__":
    # setup_logging()
    local_rank = os.environ.get("LOCAL_RANK", None)
    if local_rank == None or int(local_rank) == 0:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()

    root_logger = logging.getLogger()
    # set logging level
    if args.logging == "DEBUG":
        root_logger.setLevel(logging.DEBUG)
    elif args.logging == "INFO":
        root_logger.setLevel(logging.INFO)

    config = build_config(args, override_args)

    if os.path.exists(args.config_path):
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".yml")
        with open(temp.name, "w") as tmp:
            yaml.dump(config, tmp)
        config["task"]["wandb"].get("log_artifacts", []).append(str(temp.name))

    timestamp = torch.tensor(datetime.now().timestamp())
    timestamp = datetime.fromtimestamp(timestamp.int()).strftime("%Y-%m-%d-%H-%M-%S")

    # wandb hyperparameter sweep setup
    sweep_params = config["task"]["wandb"].get("sweep", None)

    # entrypoint if this is an agent-based sweep
    if config["task"]["wandb"]["use_wandb"] and args.sweep_id:
        wandb.agent(args.sweep_id, function=main)

    # entrypoint if we want to create a job script for submission
    elif args.job_script:
        # create JSON config file, track as artifact
        filename = os.path.join(CONFIG_PATH, "generated", f"config-{timestamp}.yml")
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with open(filename, "w") as f:
            yaml.dump(config, f)
        # create job script
        job_cls = registry.get_job_class(args.job_script)(
            config["task"]["identifier"], args.job_script
        )
        script_file = job_cls.entrypoint(
            filename, config["task"]["wandb"]["use_wandb"], args.run_mode, None
        )
        logging.info(
            f'Run the following job script to initiate task.\n"{script_file}" (copy to clipboard)'
        )

    # regular sweep entrypoint
    elif (
        config["task"]["wandb"]["use_wandb"]
        and sweep_params
        and sweep_params.get("do_sweep", False)
    ):
        sweep_path = sweep_params.get("sweep_file", None)
        with open(sweep_path, "r") as ymlfile:
            sweep_config = yaml.load(ymlfile, Loader=yaml.FullLoader)

        # update config with sweep parameters for downstream use
        config["task"]["wandb"]["sweep"]["params"] = list(
            sweep_config.get("parameters", {}).keys()
        )

        # start sweep
        sweep_id = wandb.sweep(
            sweep_config,
            project=config["task"]["wandb"].get("wandb_project", "matdeeplearn"),
        )
        sweep_count = sweep_params.get("count", 1)

        # start a sequential or parallel sweep, NOTE: sequential sweep prone to bottlenecking and early failures
        if not sweep_params.get("parallel", False):
            logging.info(
                f"Starting sequential sweep with id: {sweep_id} with max count of {sweep_count} runs."
            )
            wandb.agent(sweep_id, function=main, count=sweep_count)
        else:
            # TODO: refactor sweeps functionality to reflect job abstractions
            logging.info("Creating parallel sweeps...")
            sweep_id = "/".join(
                [
                    config["task"]["wandb"].get("wandb_entity", "fung-lab"),
                    config["task"]["wandb"].get("wandb_project", "matdeeplearn"),
                    sweep_id,
                ]
            )
            main_path = os.path.realpath(__file__)
            # find job config path if it exists
            if "slurm" in sweep_params.get("system"):
                if not sweep_params.get("job_config", None):
                    raise ValueError(
                        "Job config path not found when attempting to create parallel slurm sweep."
                    )
                job_config_path = sweep_params.get("job_config", None)
                with open(job_config_path, "r") as f:
                    job_config = yaml.safe_load(f)
                    # Start a parallel SLURM sweep task
                    script = start_sweep_tasks(
                        sweep_params.get("system"),
                        sweep_id,
                        sweep_params.get("count", 1),
                        str(args.config_path),
                        main_path,
                        job_config,
                    )
                # NOTE: Hardcoded failsafe for single GPU case, will need to be updated for distributed
                config["task"]["device"] = "cuda:0"
            elif sweep_params.get("system") == "local":
                script = start_sweep_tasks(
                    sweep_params.get("system"),
                    sweep_id,
                    sweep_params.get("count", 1),
                    str(args.config_path),
                    main_path,
                )
                # Use the automatic min allocation scheme,
                # which takes place when the training is about to start
                config["task"]["device"] = None
            else:
                raise ValueError(
                    "Invalid system type for parallel sweep. Must be either '[phoenix,perlmutter]_slurm' or 'local'."
                )
            logging.info(f"Parallel sweep script: {script}")
    else:
        main()
