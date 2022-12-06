import argparse
import copy
import logging
import pprint

from matdeeplearn.common.config.build_config import build_config
from matdeeplearn.common.config.flags import flags
from matdeeplearn.common.trainer_context import new_trainer_context
from matdeeplearn.preprocessor.processor import process_data

# import submitit

# from matdeeplearn.common.utils import setup_logging


class ChainRunner:  # submitit.helpers.Checkpointable):
    def __init__(self):
        self.config = None

    def __call__(self, config):
        # build configs for each model listed in config
        step = 0
        for model in config["models"]:

            model_args = argparse.Namespace()
            model_args.config_path = model["config_path"]
            model_args.seed = model["seed"]
            model_args.run_mode = "train"
            model_args.submit = None

            # args, override_args = parser.parse_known_args()
            model_config = build_config(model_args, [])
            print(model_config)

            if not model_config["dataset"]["processed"]:
                process_data(model_config["dataset"])

            # load previous output and include in next trainer
            if step != 0:
                # create separate directory with vasp and new scaled/feature csv - split results into individual files
                # train model 2 on same train split as model 1
                # use same validation split too

                # can either split manually, or make seed same
                pass

            # run first task
            with new_trainer_context(args=model_args, config=model_config) as ctx:
                self.config = ctx.config
                self.task = ctx.task
                self.trainer = ctx.trainer

                self.task.setup(self.trainer)

                # Print settings for job
                logging.debug("Settings: ")
                logging.debug(pprint.pformat(self.config))

                self.task.run()

            step += 1

    def checkpoint(self, *args, **kwargs):
        # new_runner = Runner()
        self.trainer.save(checkpoint_file="checkpoint.pt", training_state=True)
        self.config["checkpoint"] = self.task.chkpt_path
        self.config["timestamp_id"] = self.trainer.timestamp_id
        if self.trainer.logger is not None:
            self.trainer.logger.mark_preempting()
        # return submitit.helpers.DelayedSubmission(new_runner, self.config)


if __name__ == "__main__":
    # setup_logging()
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)

    if args.submit:  # Run on cluster
        # TODO: add setup to submit to cluster
        pass

    else:  # Run locally
        ChainRunner()(config)
