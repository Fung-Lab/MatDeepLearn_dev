
import logging
import os

class BaseTask:
    def __init__(self, config):
        self.config = config

    def setup(self, trainer):
        self.trainer = trainer
        if self.config["checkpoint"] is not None:
            self.trainer.load_checkpoint(self.config["checkpoint"])

        # save checkpoint path to runner state for slurm resubmissions
        self.chkpt_path = os.path.join(
            self.trainer.config["cmd"]["checkpoint_dir"], "checkpoint.pt"
        )

    def run(self):
        raise NotImplementedError
