import argparse
from pathlib import Path


class Flags:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="MatDeepLearn inputs")
        self.add_core_args()

    def get_parser(self):
        return self.parser

    def add_core_args(self):
        self.parser.add_argument(
            "--config_path",
            required=True,
            default="configs/config.yml",
            type=Path,
            help="Path to a config file with data and model parameters (default: config.yml)",
        )
        self.parser.add_argument(
            "--run_mode",
            choices=["train", "predict", "finetune"],
            required=False,
            type=str,
            help="Choices for run modes: Training, Predict, Repeat, CV, Hyperparameter, Ensemble, Analysis",
        )
        # self.parser.add_argument(
        #     "--job_name",
        #     default=None,
        #     type=str,
        #     help="name of your job and output files/folders",
        # )
        # self.parser.add_argument(
        #     "--model",
        #     default=None,
        #     type=str,
        #     help="CGCNN_demo, MPNN_demo, SchNet_demo, MEGNet_demo, GCN_demo, SOAP_demo, SM_demo",
        # )
        self.parser.add_argument(
            "--seed",
            type=int,
            help="seed for data split, 0=random",
        )
        # parser.add_argument(
        #     "--model_path",
        #     default=None,
        #     type=str,
        #     help="path of the model .pth file",
        # )
        # parser.add_argument(
        #     "--save_model",
        #     default=None,
        #     type=str,
        #     help="Save model",
        # )
        # parser.add_argument(
        #     "--load_model",
        #     default=None,
        #     type=str,
        #     help="Load model",
        # )
        # parser.add_argument(
        #     "--write_output",
        #     default=None,
        #     type=str,
        #     help="Write outputs to csv",
        # )
        # parser.add_argument(
        #     "--parallel",
        #     default=None,
        #     type=str,
        #     help="Use parallel mode (ddp) if available",
        # )
        # parser.add_argument(
        #     "--reprocess",
        #     default=None,
        #     type=str,
        #     help="Reprocess data since last run",
        # )
        # ###Processing arguments
        #
        # #TODO: remove data and model arguments from here (loaded in config file)
        #
        # parser.add_argument(
        #     "--data_path",
        #     default=None,
        #     type=str,
        #     help="Location of data containing structures (json or any other valid format) and accompanying files",
        # )
        # parser.add_argument("--format", default=None, type=str, help="format of input data")
        # ###Training arguments
        # parser.add_argument("--train_ratio", default=None, type=float, help="train ratio")
        # parser.add_argument(
        #     "--val_ratio", default=None, type=float, help="validation ratio"
        # )
        # parser.add_argument("--test_ratio", default=None, type=float, help="test ratio")
        # parser.add_argument(
        #     "--verbosity", default=None, type=int, help="prints errors every x epochs"
        # )
        # parser.add_argument(
        #     "--target_index",
        #     default=None,
        #     type=int,
        #     help="which column to use as target property in the target file",
        # )
        # ###Model arguments
        # parser.add_argument(
        #     "--epochs",
        #     default=None,
        #     type=int,
        #     help="number of total epochs to run",
        # )
        # parser.add_argument("--batch_size", default=None, type=int, help="batch size")
        # parser.add_argument("--lr", default=None, type=float, help="learning rate")

        # TODO: add cluster args
        self.parser.add_argument(
            "--submit", action="store_true", help="Submit job to cluster"
        )
        # TODO: add checkpoint arg
        # TODO: timestamp id arg?


flags = Flags()
