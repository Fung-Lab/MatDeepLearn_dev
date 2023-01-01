import logging
import os

from matdeeplearn.common.registry import registry

"""
These classes are used for running with a config file via command line
"""


class BaseTask:
    def __init__(self, config):
        self.config = config

    def setup(self, trainer):
        self.trainer = trainer
        use_checkpoint = self.config["model"]["load_model"]
        if use_checkpoint:
            logging.info("Attempting to load most recent checkpoint...")
            self.trainer.load_checkpoint(most_recent=True)
            logging.info("Recent checkpoint loaded successfully.")
        # save checkpoint path to runner state for slurm resubmissions
        # self.chkpt_path = os.path.join(
        #     self.trainer.config["cmd"]["checkpoint_dir"], "checkpoint.pt"
        # )

    def run(self):
        raise NotImplementedError


@registry.register_task("train")
class TrainTask(BaseTask):
    def _process_error(self, e: RuntimeError):
        e_str = str(e)
        if (
            "find_unused_parameters" in e_str
            and "torch.nn.parallel.DistributedDataParallel" in e_str
        ):
            for name, parameter in self.trainer.model.named_parameters():
                if parameter.requires_grad and parameter.grad is None:
                    logging.warning(
                        f"Parameter {name} has no gradient. Consider removing it from the model."
                    )

    def run(self):
        try:
            self.trainer.train()
        except RuntimeError as e:
            self._process_error(e)
            raise e
