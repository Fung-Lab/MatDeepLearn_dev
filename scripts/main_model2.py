import logging
import os
import pprint

import torch
from torch import distributed as dist

from matdeeplearn.common.config.build_config import build_config
from matdeeplearn.common.config.flags import flags
from matdeeplearn.common.trainer_context import new_trainer_context
from matdeeplearn.preprocessor.processor import process_data

# import submitit

# from matdeeplearn.common.utils import setup_logging


class Runner2:  # submitit.helpers.Checkpointable):
    def __init__(self):
        self.config = None

    def __call__(self, config):
        with new_trainer_context(args=args, config=config) as ctx:
            self.config = ctx.config
            self.task = ctx.task
            self.trainer = ctx.trainer

            model_path = "/global/cfs/projectdirs/m3641/Sarah/results/2023-06-21-19-25-04-CGCNN_MP_DOS_node_level_prediction/checkpoint/best_checkpoint.pt"
            load_model = torch.load(model_path, map_location=self.trainer.device)
            load_state = load_model["state_dict"]

            model_state = self.trainer.model.state_dict()

            logging.debug(model_state.keys())
            for name, param in load_state.items():
                model_state[name].copy_(param)

            logging.info("loaded pretrained CGCNN model")

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


if __name__ == "__main__":

    # setup_logging()
    local_rank = os.environ.get("LOCAL_RANK", None)
    if local_rank is None or int(local_rank) == 0:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)

    if not config["dataset"]["processed"]:
        process_data(config["dataset"])

    if args.submit:  # Run on cluster
        # TODO: add setup to submit to cluster
        pass

    else:  # Run locally
        Runner2()(config)
