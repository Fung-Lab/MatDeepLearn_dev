import logging

from matdeeplearn.common.registry import registry
"""
These classes are used for running with a config file via command line
"""


class BaseTask:
    def __init__(self, config):
        self.config = config
        
    def setup(self, trainer):
        self.trainer = trainer
        use_checkpoint = self.config["model"].get("load_model", False)
        if use_checkpoint:
            logging.info("Attempting to load most recent checkpoint...")
            self.trainer.load_checkpoint()
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


@registry.register_task("predict")
class PredictTask(BaseTask):
    def run(self):
        assert (
            self.trainer.data_loader.get("predict_loader") is not None
        ), "Predict dataset is required for making predictions"
        assert self.config["task"][
            "checkpoint_dir"
        ], "Specify checkpoint directory for loading the model"
        assert self.config["model"][
            "load_model"
        ], "Set load_model = True to use the model for prediction"    
        results_dir = f"predictions/{self.config['dataset']['name']}"
        try:
            self.trainer.predict(
                loader=self.trainer.data_loader["predict_loader"], split="predict", results_dir=results_dir
            )
        except RuntimeError as e:
            logging.warning("Errors in predict task")
            raise e
