import yaml
import logging
from matdeeplearn.trainers import PropertyTrainer
from matdeeplearn.preprocessor.processor import process_data

#set up logging, otherwise nothing will be printed in the terminal
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

#load the config file
with open("configs/config.yml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)

#Alternatively, define the config here as a dict
config = {
  "trainer": "property",
  "task": {
    "run_mode": "train",
    "identifier": "my_train_job",
    "parallel": False,
    "seed": 2445,
    "save_dir": None,
    "continue_job": False,
    "load_training_state": False,
    "checkpoint_path": None,
    "write_output": True
  },
  "model": {
    "name": "CGCNN",
    "save_model": True,
    "model_path": "my_model.pth",
    "edge_steps": 50,
    "self_loop": True,
    "dim1": 100,
    "dim2": 150,
    "pre_fc_count": 1,
    "gc_count": 4,
    "post_fc_count": 3,
    "pool": "global_mean_pool",
    "pool_order": "early",
    "batch_norm": True,
    "batch_track_stats": True,
    "act": "relu",
    "dropout_rate": 0
  },
  "optim": {
    "max_epochs": 10,
    "max_checkpoint_epochs": 0,
    "lr": 0.002,
    "loss": {
      "loss_type": "TorchLossWrapper",
      "loss_args": {
        "loss_fn": "l1_loss"
      }
    },
    "batch_size": 100,
    "optimizer": {
      "optimizer_type": "AdamW",
      "optimizer_args": {}
    },
    "scheduler": {
      "scheduler_type": "ReduceLROnPlateau",
      "scheduler_args": {
        "mode": "min",
        "factor": 0.8,
        "patience": 10,
        "min_lr": 0.00001,
        "threshold": 0.0002
      }
    },
    "verbosity": 5
  },
  "dataset": {
    "name": "test_data",
    "processed": False,
    "src": "data/test_data/data_graph_scalar.json",
    "target_path": None,
    "pt_path": "data/",
    "prediction_level": "graph",
    "transforms": [
      {
        "name": "GetY",
        "args": {
          "index": -1
        },
        "otf": True
      }
    ],
    "all_neighbors": True,
    "edge_calc_method": "ocp",
    "data_format": "json",
    "node_representation": "onehot",
    "additional_attributes": [],
    "verbose": True,
    "preprocess_params": {
      "cutoff_radius": 8,
      "n_neighbors": 250,
      "num_offsets": 1,
      "edge_steps": 50
    },
    "train_ratio": 0.85,
    "val_ratio": 0.05,
    "test_ratio": 0.1
  }
}

#Process the data according to config
process_data(config["dataset"])

#Set up trainer from config
trainer = PropertyTrainer.from_config(config)

#Train the model
trainer.train()

#Use the model to perform a prediction on the test dataset
out = trainer.predict(loader=trainer.data_loader["test_loader"], split="predict", results_dir="test")
print(out["ids"][0:10], out["predict"][0:10], out["target"][0:10])

#Load a model and perform predict
trainer.checkpoint_path = "results/2023-05-25-21-08-16-my_train_job/checkpoint/best_checkpoint.pt"
trainer.load_checkpoint(config["task"].get("load_training_state", True))
out = trainer.predict(loader=trainer.data_loader["test_loader"], split="predict", results_dir="test")
print(out["ids"][0:10], out["predict"][0:10], out["target"][0:10])






