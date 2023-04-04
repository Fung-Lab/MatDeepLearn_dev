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

# import submitit

# from matdeeplearn.common.utils import setup_logging

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.autograd.set_detect_anomaly(True)


class Runner:  # submitit.helpers.Checkpointable):
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
        # return submitit.helpers.DelayedSubmission(new_runner, self.config)


def get_nested(data, *args):
    if args and data:
        element = args[0]
        if element and element in data:
            value = data.get(element)
            return value if len(args) == 1 else get_nested(value, *args[1:])


def flatten(my_dict):
    result = {}
    for key, value in my_dict.items():
        if isinstance(value, dict):
            result.update(flatten(value))
        else:
            result[key] = value
    return result


def wandb_setup(config):
    metadata = config["task"]["wandb"].get("metadata", {})
    _wandb_config = {
        "metadata": metadata,
    }

    timestamp = torch.tensor(datetime.now().timestamp())

    # wandb hyperparameter sweep setup
    sweep_config = config["task"]["wandb"].get("sweep_config", None)

    # TODO: finish sweep integration
    if sweep_config and sweep_config.get("sweep", False):
        sweep_path = sweep_config.get("sweep_path", None)
        with open(sweep_path, "r") as ymlfile:
            sweep_config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        params = sweep_config.get("parameters", {})
        sweep_id = wandb.sweep(
            sweep_config,
            project=config["task"]["wandb"].get("wandb_project", "matdeeplearn"),
        )
        sweep_count = sweep_config.get("count", 3)
        logging.info(
            f"Starting sweep with id: {sweep_id} with max count of {sweep_count} runs."
        )
        wandb.agent(sweep_id, function=Runner(), count=sweep_count)

    # update config with chosen parameters from config (to avoid clutter)
    track_params = [
        p.split(".") for p in config["task"]["wandb"].get("track_params", {})
    ]
    track_params = {p[-1]: get_nested(config, *p) for p in track_params}
    # transform hyperparams
    transforms = {
        key: val
        for d in config["dataset"]["transforms"]
        for key, val in flatten(d).items()
        if key != "name" and key != "otf"
    }

    _wandb_config.update(transforms)
    _wandb_config.update(track_params)

    timestamp = datetime.fromtimestamp(timestamp.int()).strftime("%Y-%m-%d-%H-%M-%S")

    if config["task"]["run_id"] != "" and not config["model"]["load_model"]:
        raise ValueError("Must load from checkpoint if also resuming wandb run.")

    wandb.init(
        settings=wandb.Settings(start_method="fork"),
        project=config["task"]["wandb"].get("wandb_project", "matdeeplearn"),
        entity=config["task"]["wandb"].get("wandb_entity", "fung-lab"),
        name=f"{timestamp}-{config['task']['identifier']}",
        notes=config["task"]["wandb"].get("notes", None),
        tags=config["task"]["wandb"].get("tags", None),
        config=_wandb_config,
        id=config["task"]["run_id"] if config["task"]["run_id"] != "" else None,
        resume="must" if config["task"]["run_id"] != "" else None,
    )

    wandb_artifacts = config["task"]["wandb"].get("log_artifacts", [])

    # create wandb artifacts
    for _, artifact in enumerate(wandb_artifacts):
        if not os.path.exists(artifact["path"]):
            raise ValueError(
                f"Artifact {artifact['path']} does not exist. Please check the path."
            )
        wandb.save(artifact["path"])


def main():
    """Entrypoint for MatDeepLearn inference tasks

    Args:
        argv (dict): main function arguments
    """
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

    # Run locally if we do not perform a sweep
    elif not config["task"]["wandb"]["sweep"]["do_sweep"]:
        Runner()(config, args)


if __name__ == "__main__":
    main()
