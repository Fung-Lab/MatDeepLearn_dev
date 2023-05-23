import logging
import pprint

from matdeeplearn.common.config.build_config import build_config
from matdeeplearn.common.config.flags import flags
from matdeeplearn.common.trainer_context import new_trainer_context
from matdeeplearn.preprocessor.processor import process_data
import wandb

# import submitit

# from matdeeplearn.common.utils import setup_logging
sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters': 
    {
        'lr': {'max': .005, 'min': .00001},
        'dim1': {'values': [64, 100, 128, 200, 256, 512]},
        'dim2': {'values': [64, 100, 128, 200, 256, 512]},
        'gc_count': {'values': [1, 2, 3, 4, 5, 6]},
        'dropout_rate': {'values': [0, .05, .1, .15, .2, .25]},
        'cutoff_radius': {'max': 12.0, 'min': 4.0},
        'edge_steps': {'values': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]},
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project='my-first-sweep')

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
args = None

#if __name__ == "__main__":
def main():
    # setup_logging()
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)

    if not config["dataset"]["processed"]:
        process_data(config["dataset"])

    run = wandb.init()
    config["optim"]["lr"] = wandb.config.lr
    config["model"]["dim1"] = wandb.config.dim1
    config["model"]["dim2"] = wandb.config.dim2
    config["model"]["gc_count"] = wandb.config.gc_count
    config["model"]["dropout_rate"] = wandb.config.dropout_rate
    config["dataset"]["preprocess_params"]["cutoff_radius"] = wandb.config.cutoff_radius
    config["dataset"]["preprocess_params"]["edge_steps"] = wandb.config.edge_steps

    if args.submit:  # Run on cluster
        # TODO: add setup to submit to cluster
        pass

    else:  # Run locally
        Runner()(config)
wandb.agent(sweep_id, function=main, count=100)

#if __name__ == "__main__":
#    main()
