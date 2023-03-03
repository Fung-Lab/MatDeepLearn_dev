import argparse
import copy
import logging
import pprint

import torch

from matdeeplearn.common.config.build_config import build_config
from matdeeplearn.common.config.flags import flags
from matdeeplearn.common.data import dataset_split, get_dataloader, get_dataset
from matdeeplearn.common.trainer_context import new_trainer_context
from matdeeplearn.preprocessor.processor import process_data

# from matdeeplearn.preprocessor.datasets import ChainDataset
# import submitit

# from matdeeplearn.common.utils import setup_logging


class ChainRunner:  # submitit.helpers.Checkpointable):
    def __init__(self):
        self.config = None

    def __call__(self, config):
        # build configs for each model listed in config
        step = 0
        train_data, val_data, test_data = None, None, None
        for model in config["models"]:

            model_args = argparse.Namespace()
            model_args.config_path = model["config_path"]
            model_args.seed = model["seed"]
            model_args.run_mode = "train"
            model_args.submit = None
            print(model_args)

            # args, override_args = parser.parse_known_args()
            model_config = build_config(model_args, [])
            print(model_config)
            dataset_config = model_config["dataset"]
            if not model_config["dataset"]["processed"]:
                process_data(dataset_config)

            # load previous output and include in next trainer
            if step == 0:
                dataset = get_dataset(
                    dataset_config["pt_path"], dataset_config.get("target_index", 0)
                )
                train_data, val_data, test_data = dataset_split(
                    dataset,
                    dataset_config["train_ratio"],
                    dataset_config["val_ratio"],
                    dataset_config["test_ratio"],
                )

            # run first task
            with new_trainer_context(args=model_args, config=model_config) as ctx:
                self.config = ctx.config
                self.task = ctx.task
                self.trainer = ctx.trainer

                # subsequent steps, can use predicted data
                if step != 0:
                    pass

                self.task.setup(self.trainer)

                # Print settings for job
                logging.debug("Settings: ")
                logging.debug(pprint.pformat(self.config))

                model = self.task.run()

            # set scaled and scaling_factor in out
            # save data to file for next processing

            # after model 1 has run, get model prediction and use for prediction in model 2
            if step == 0:
                model1_out = self.trainer.predict(
                    self.trainer.test_loader, split="model1_test"
                )
                # print(test_data[0])
                # print(test_data[0].scaled)
                # print(len(model1_out))
                # print(model1_out[0])
                # print(model1_out[0][0])
                # print(model1_out[0][0].shape)
                # print(model1_out[0][1])
                # print(model1_out[0][1].shape)
                # exit()

            step += 1

        # after model 2 runs, use model1_out in model2 predictions (model1_out is (scaled, scaling_factor))
        counter = 0
        new_data_lst = []
        print(len(test_data.indices))
        print(len(model1_out))
        # exit()
        for i in test_data.indices:
            # for i in range(55):
            # print(model1_out[counter])
            new_data = dataset.__setitem__(i, model1_out[counter])
            new_data_lst.append(new_data)
            counter += 1

        # create new Dataset from data list
        # new_dataset = ChainDataset(new_data_lst)
        print(new_data_lst[0])
        print(new_data_lst[0].scaled)
        print(new_data_lst[0].structure_id)
        print(test_data[0])
        print(test_data[0].scaled)
        print(test_data[0].structure_id)
        # exit()
        test_loader = get_dataloader(
            new_data_lst, batch_size=model_config["optim"]["batch_size"], sampler=None
        )
        self.trainer.test_loader = test_loader

        self.trainer.predict(self.trainer.test_loader, split="model2_test")

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
