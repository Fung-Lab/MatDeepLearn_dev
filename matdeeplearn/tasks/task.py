import logging

from matdeeplearn.common.registry import registry
"""
These classes are used for running with a config file via command line
"""


class BaseTask:
    def __init__(self, config):
        self.config = config

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

    def setup(self, trainer):
        self.trainer = trainer 
        use_checkpoint = self.config["task"].get("continue_job", False)
        if use_checkpoint:
            logging.info("Attempting to load checkpoint...")
            self.trainer.load_checkpoint(self.config["task"].get("load_training_state", True))
            logging.info("Recent checkpoint loaded successfully.")
            
        # save checkpoint path to runner state for slurm resubmissions
        # self.chkpt_path = os.path.join(
        #     self.trainer.config["cmd"]["checkpoint_dir"], "checkpoint.pt"
        # )
        
    def run(self):
        try:
            self.trainer.train()     
            
        except RuntimeError as e:
            self._process_error(e)
            raise e


@registry.register_task("predict")
class PredictTask(BaseTask):
    def setup(self, trainer):
        self.trainer = trainer       
        assert self.config["task"][
            "checkpoint_path"
        ], "Specify checkpoint directory for loading the model"         
        logging.info("Attempting to load checkpoint...")
        self.trainer.load_checkpoint(self.config["task"].get("load_training_state", True))
        logging.info("Recent checkpoint loaded successfully.")

    def run(self):
        # if isinstance(self.trainer.data_loader, list):
        assert (
            self.trainer.data_loader[0].get("predict_loader") is not None
        ),  "Predict dataset is required for making predictions"
        # else:
            # assert (
                # self.trainer.data_loader.get("predict_loader") is not None
            # ),  "Predict dataset is required for making predictions"
        results_dir = f"predictions/{self.config['dataset']['name']}"
        try:
            # if isinstance(self.trainer.data_loader, list):
            self.trainer.predict(
                loader=self.trainer.data_loader, split="predict", results_dir=results_dir, labels=self.config["task"]["labels"],
            )
            # else:
                # self.trainer.predict(
                    # loader=self.trainer.data_loader["predict_loader"], split="predict", results_dir=results_dir, labels=self.config["task"]["labels"],
                # )
        except RuntimeError as e:
            logging.warning("Errors in predict task")
            raise e


@registry.register_task("finetune")
class FineTuneTask(BaseTask):
    def setup(self, trainer):
        self.trainer = trainer   
        assert self.config["task"][
            "checkpoint_path"
        ], "Specify checkpoint directory for loading the model"     
        logging.info("Attempting to load checkpoint...")
        self.trainer.load_pre_trained_weights(self.config["task"].get("load_training_state", False))
        logging.info("Pretrained model loaded successfully.")
        
    def run(self):
        try:
            self.trainer.train()     
            
        except RuntimeError as e:
            self._process_error(e)
            raise e
