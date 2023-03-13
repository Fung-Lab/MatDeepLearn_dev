import logging
import os
import pprint
from datetime import datetime

import torch
import wandb

from matdeeplearn.common.config.build_config import build_config
from matdeeplearn.common.config.flags import flags
from matdeeplearn.common.trainer_context import new_trainer_context
from matdeeplearn.preprocessor.processor import process_data

# import submitit

# from matdeeplearn.common.utils import setup_logging

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.autograd.set_detect_anomaly(True)


class Runner:  # submitit.helpers.Checkpointable):
    def __init__(self):
        self.config = None

    def __call__(self, config):
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
        # return submitit.helpers.DelayedSubmission(new_runner, self.config)


def wandb_setup(config):
    metadata = config["task"]["wandb"].get("metadata", {})
    _wandb_config = {
        "metadata": metadata,
    }

    timestamp = torch.tensor(datetime.now().timestamp())

    # wandb hyperparameter sweep setup
    # sweep_config = config["task"]["wandb"]["sweep_config"]

    # if sweep_config and sweep_config.get("sweep", False):
    #     sweep_path = sweep_config.get("sweep_path", None)
    #     with open(sweep_path, "r") as ymlfile:
    #         sweep_config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    #     params = sweep_config.get("parameters", {})

    #     sweep_id = wandb.sweep(
    #         sweep_config,
    #         project=config["task"]["wandb"].get("wandb_project", "matdeeplearn"),
    #     )

    # update config with model hyperparams
    _wandb_config.update(config["model"]["hyperparams"])
    # update config with processing hyperparams
    _wandb_config.update({"transforms": config["dataset"]["transforms"]})

    timestamp = datetime.fromtimestamp(timestamp.int()).strftime("%Y-%m-%d-%H-%M-%S")

    wandb.init(
        settings=wandb.Settings(start_method="fork"),
        project=config["task"]["wandb"].get("wandb_project", "matdeeplearn"),
        entity=config["task"]["wandb"].get("wandb_entity", "fung-lab"),
        name=f"{timestamp}-{config['task']['identifier']}",
        notes=config["task"]["wandb"].get("notes", None),
        tags=config["task"]["wandb"].get("tags", None),
        config=_wandb_config,
    )

    wandb_artifacts = config["task"]["wandb"].get("log_artifacts", [])

    # create wandb artifacts
    for i, artifact in enumerate(wandb_artifacts):
        if not os.path.exists(artifact["path"]):
            raise ValueError(
                f"Artifact {artifact['path']} does not exist. Please check the path."
            )
        wandb.save(artifact["path"])


if __name__ == "__main__":
    # setup_logging()
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)

    # wandb setup process, initiate a run
    if args.use_wandb:
        wandb_setup(config)

    # override config from CLI, useful for quick experiments/debugging purposes
    config["task"]["wandb"]["use_wandb"] = (
        args.use_wandb and config["task"]["wandb"]["use_wandb"]
    )

    if not config["dataset"]["processed"]:
        process_data(config["dataset"], config["task"]["wandb"])

    if args.submit:  # Run on cluster
        # TODO: add setup to submit to cluster
        pass

    else:  # Run locally
        Runner()(config)
