import yaml
import pytest

from matdeeplearn.preprocessor.processor import process_data

from utils import trainer_property, trainer_context, assert_valid_predictions

@pytest.fixture
def params():
    with open("test/configs/cpu/test_training.yml", "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    config["task"]["run_mode"] = "train"
    return config

@pytest.fixture
def models():
    return ["CGCNN", "SchNet"]

def test_training(params, models):
    process_data(params["dataset"])
 
    trainers = []
    for model in models:
        params["model"]["name"] = model
        trainers.append(trainer_context(params, True))
        trainers.append(trainer_property(params, True))

    for trainer in trainers:
        assert_valid_predictions(trainer, "test_loader")
