import yaml
import pytest

from matdeeplearn.preprocessor.processor import process_data

from utils import trainer_property, trainer_context, assert_valid_predictions


@pytest.fixture
def params():
    with open("test/configs/cpu/test_predict.yml", "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    config["task"]["run_mode"] = "predict"
    return config

def test_predict(params):
    process_data(params["dataset"])

    trainers = []
    trainers.append(trainer_property(params, False))
    trainers.append(trainer_context(params, False))

    for trainer in trainers:
        assert_valid_predictions(trainer, "predict_loader")
