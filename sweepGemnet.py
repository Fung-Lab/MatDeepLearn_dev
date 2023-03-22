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
        'lr': {'max': .003, 'min': .00001},
        'emb_size': {'values': [32, 64, 128, 256, 512]},
        'emb_size_trip': {'values': [32, 64, 128, 256, 512]},
        'emb_size_rbf': {'values': [4, 8, 16, 32]},
        'emb_size_bil_trip': {'values': [8, 16, 32, 64]},
        'num_atom' : {'values': [1, 2, 3, 4]},
        'num_atom_emb' : {'values': [0, 1, 2, 3, 4]},
        'num_blocks': {'values': [2, 4, 6, 8]},
        'num_radial': {'values': [4, 6, 8, 10]},
        'num_spherical': {'values': [4, 6, 8, 10]},
        'cutoff': {'max': 12.0, 'min': 4.0},
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
    config["model"]["emb_size_atom"] = wandb.config.emb_size
    config["model"]["emb_size_edge"] = wandb.config.emb_size
    config["model"]["emb_size_trip_in"] = wandb.config.emb_size

    config["model"]["emb_size_trip_out"] = wandb.config.emb_size_trip
    config["model"]["emb_size_trip"] = wandb.config.emb_size_trip
    config["model"]["emb_size_quad_in"] = wandb.config.emb_size_trip
    config["model"]["emb_size_quad_out"] = wandb.config.emb_size_trip
    config["model"]["emb_size_aint_in"] = wandb.config.emb_size_trip
    config["model"]["emb_size_aint_out"] = wandb.config.emb_size_trip

    config["model"]["emb_size_rbf"] = wandb.config.emb_size_rbf
    config["model"]["emb_size_cbf"] = wandb.config.emb_size_rbf
    config["model"]["emb_size_sbf"] = wandb.config.emb_size_rbf
    config["model"]["emb_size_bil_trip"] = wandb.config.emb_size_bil_trip

    config["model"]["num_atom"] = wandb.config.num_atom
    config["model"]["num_atom_emb"] = wandb.config.num_atom_emb
    

    config["model"]["num_blocks"] = wandb.config.num_blocks
    config["model"]["num_radial"] = wandb.config.num_radial
    config["model"]["num_spherical"] = wandb.config.num_spherical

    config["model"]["cutoff"] = wandb.config.cutoff

    if args.submit:  # Run on cluster
        # TODO: add setup to submit to cluster
        pass

    else:  # Run locally
        Runner()(config)
wandb.agent(sweep_id, function=main, count=50)

#if __name__ == "__main__":
#    main()
