import argparse
import copy
import logging
import pprint

import torch
from torch_geometric.data import Data, InMemoryDataset

from matdeeplearn.common.config.build_config import build_config
from matdeeplearn.common.config.flags import flags
from matdeeplearn.common.data import dataset_split, get_dataloader, get_dataset
from matdeeplearn.common.trainer_context import new_trainer_context
from matdeeplearn.preprocessor.processor import process_data

# from matdeeplearn.preprocessor.datasets import ChainDataset
# import submitit

# from matdeeplearn.common.utils import setup_logging


class ChainRunner3:  # submitit.helpers.Checkpointable):
    def __init__(self):
        self.config = None

    def __call__(self, config):

        # build configs for each model listed in config
        # train_data, val_data, test_data = None, None, None
        model_config_list = []
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
            model_config_list.append((model_config, model_args))

        model_config_1, model_args_1 = model_config_list[0]
        model_config_2, model_args_2 = model_config_list[1]

        # process data for first model - same dataset will be used in second model
        # start with same dataset split
        dataset_config = model_config_1["dataset"]
        if not model_config_1["dataset"]["processed"]:
            process_data(dataset_config)

        dataset = get_dataset(
            dataset_config["pt_path"], dataset_config.get("target_index", 0)
        )
        train_data, val_data, test_data = dataset_split(
            dataset,
            dataset_config["train_ratio"],
            dataset_config["val_ratio"],
            dataset_config["test_ratio"],
        )

        # initialize training for model 1
        # run first task
        with new_trainer_context(args=model_args_1, config=model_config_1) as ctx:
            self.config = ctx.config
            self.task = ctx.task
            self.trainer = ctx.trainer

            # Print settings for job
            logging.debug("Settings: ")
            logging.debug(pprint.pformat(self.config))

            # swap out init dataloaders for dataloaders with same split (but with shuffle)
            batch_size = model_config_1["optim"]["batch_size"]
            train_loader = get_dataloader(train_data, batch_size=batch_size)
            val_loader = get_dataloader(val_data, batch_size=batch_size)
            test_loader = get_dataloader(test_data, batch_size=batch_size)

            self.trainer.train_loader = train_loader
            self.trainer.val_loader = val_loader
            self.trainer.test_loader = test_loader

            # train model 1
            # model = self.task.run()
            # model_path = "/Users/seisenach3/Documents/Georgia Tech/Fung Lab/Results/2023-05-24-23-02-40-DOSPredict_MP_DOS/best_checkpoint.pt"
            model_path = "/global/cfs/projectdirs/m3641/Sarah/results/2023-05-24-23-02-40-DOSPredict_MP_DOS/checkpoint/best_checkpoint.pt"
            load_model = torch.load(model_path, map_location=self.trainer.device)
            load_state = load_model["state_dict"]

            model_state = self.trainer.model.state_dict()

            print(model_state.keys())
            for name, param in load_state.items():
                model_state[name].copy_(param)

            print("loaded pretrained model")

            self.task.setup(self.trainer)

            # make dataloaders with no shuffle but same split
            train_loader_unshuffle = get_dataloader(
                train_data, batch_size=batch_size, shuffle=False
            )
            val_loader_unshuffle = get_dataloader(
                val_data, batch_size=batch_size, shuffle=False
            )
            test_loader_unshuffle = get_dataloader(
                test_data, batch_size=batch_size, shuffle=False
            )

            # predict model 1 output using new dataloaders
            model1_train_out, _ = self.trainer.predict(
                train_loader_unshuffle, split="model1_predict_train"
            )
            model1_val_out, _ = self.trainer.predict(
                val_loader_unshuffle, split="model1_predict_val"
            )
            model1_test_out, _ = self.trainer.predict(
                test_loader_unshuffle, split="model1_predict_test"
            )

        # swap out data from dataset split with predicted data
        new_data_list = []
        for count, idx in enumerate(train_data.indices):
            new_train_data = dataset.__setitem__(idx, model1_train_out[count])
            new_data_list.append(new_train_data)

        for count, idx in enumerate(val_data.indices):
            new_val_data = dataset.__setitem__(idx, model1_val_out[count])
            new_data_list.append(new_val_data)

        for count, idx in enumerate(test_data.indices):
            new_test_data = dataset.__setitem__(idx, model1_test_out[count])
            new_data_list.append(new_test_data)

        # save data
        data, slices = InMemoryDataset.collate(new_data_list)
        save_path = "/Users/seisenach3/Documents/Georgia Tech/Fung Lab/MatDeepLearn/data/dos_chain_predicted/data.pt"
        "/global/cfs/projectdirs/m3641/Sarah/datasets/MP_DOS_data/chain_predictions/data.pt"
        torch.save((data, slices), save_path)

        # run second task with saved data
        with new_trainer_context(args=model_args_2, config=model_config_2) as ctx:
            self.config = ctx.config
            self.task = ctx.task
            self.trainer = ctx.trainer

            self.task.setup(self.trainer)

            # Print settings for job
            logging.debug("Settings: ")
            logging.debug(pprint.pformat(self.config))

            self.task.run()

            _, predict_error = self.trainer.predict(
                self.trainer.test_loader, split="model2_test"
            )

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
    root_logger.setLevel(logging.DEBUG)

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)

    if args.submit:  # Run on cluster
        # TODO: add setup to submit to cluster
        pass

    else:  # Run locally
        ChainRunner3()(config)
