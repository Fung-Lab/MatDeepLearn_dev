import copy
import logging

# import submitit

from matdeeplearn.common.config.flags import flags
from matdeeplearn.common.config.build_config import build_config
from matdeeplearn.common.trainer_context import new_trainer_context
# from matdeeplearn.common.utils import setup_logging

from matdeeplearn.process.processor import process_data


class Runner(): #submitit.helpers.Checkpointable):
    def __init__(self):
        self.config = None

    def __call__(self, config):
        with new_trainer_context(args=args, config=config) as ctx:
            self.config = ctx.config
            self.task = ctx.task
            self.trainer = ctx.trainer

            print(self.trainer.dataset[0])

            input()

            self.task.setup(self.trainer)
            # self.task.run()

    def checkpoint(self, *args, **kwargs):
        new_runner = Runner()
        self.trainer.save(checkpoint_file="checkpoint.pt", training_state=True)
        self.config["checkpoint"] = self.task.chkpt_path
        self.config["timestamp_id"] = self.trainer.timestamp_id
        if self.trainer.logger is not None:
            self.trainer.logger.mark_preempting()
        return submitit.helpers.DelayedSubmission(new_runner, self.config)


if __name__ == "__main__":
    # setup_logging()

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)

    if not config["dataset"]["processed"]:
        process_data(config["dataset"])

    print(config)


    if args.submit:  # Run on cluster
        #TODO: add setup to submit to cluster
        pass

    else:  # Run locally
        Runner()(config)
