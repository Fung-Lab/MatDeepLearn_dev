import logging
import os
import pprint
from datetime import datetime

import torch
import wandb
import yaml

from matdeeplearn.common.config.build_config import build_config
from matdeeplearn.common.config.flags import flags
from matdeeplearn.common.trainer_context import new_trainer_context
from matdeeplearn.preprocessor.processor import process_data
from matdeeplearn.common.utils import DictTools

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.autograd.set_detect_anomaly(True)


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

    if config["task"]["run_id"] != "" and not config["model"]["load_model"]:
        raise ValueError("Must load from checkpoint if also resuming wandb run.")

    timestamp = torch.tensor(datetime.now().timestamp())
    timestamp = datetime.fromtimestamp(timestamp.int()).strftime("%Y-%m-%d-%H-%M-%S")

    wandb_id = config["task"]["run_id"] if config["task"]["run_id"] != "" else None
    wandb.init(
        settings=wandb.Settings(start_method="fork"),
        project=config["task"]["wandb"].get("wandb_project", "matdeeplearn"),
        entity=config["task"]["wandb"].get("wandb_entity", "fung-lab"),
        name=f"{timestamp}-{config['task']['identifier']}",
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
            if not os.path.exists(artifact):
                raise ValueError(
                    f"Artifact {artifact} does not exist. Please check the path."
                )
            else:
                print(artifact, type(artifact))
                wandb.save(artifact)
    else:
        wandb.run.name = f"{wandb.run.name}-{wandb.run.id}"


def main():
    """Entrypoint for MatDeepLearn inference tasks"""
    wandb_setup(config)
    if not config["dataset"]["processed"]:
        process_data(config["dataset"], config["task"]["wandb"])
    Runner()(config, args)


if __name__ == "__main__":
    # setup_logging()
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)
    
    # add config path as an artifact manually
    config["task"]["wandb"].get("log_artifacts", []).append(str(args.config_path))

    # override config from CLI, useful for quick experiments/debugging purposes
    config["task"]["wandb"]["use_wandb"] = (
        args.use_wandb and config["task"]["wandb"]["use_wandb"]
    )
    # wandb hyperparameter sweep setup
    sweep_params = config["task"]["wandb"].get("sweep", None)

    if args.use_wandb and sweep_params and sweep_params.get("do_sweep", False):
        sweep_path = sweep_params.get("sweep_file", None)
        with open(sweep_path, "r") as ymlfile:
            sweep_config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        # udpate config with sweep parameters for downstream use
        config["task"]["wandb"]["sweep"]["params"] = list(
            sweep_config.get("parameters", {}).keys()
        )
        # start sweep
        sweep_id = wandb.sweep(
            sweep_config,
            project=config["task"]["wandb"].get("wandb_project", "matdeeplearn"),
        )
        sweep_count = sweep_params.get("count", 1)
        logging.info(
            f"Starting sweep with id: {sweep_id} with max count of {sweep_count} runs."
        )
        wandb.agent(sweep_id, function=main, count=sweep_count)
    else:
        main()
