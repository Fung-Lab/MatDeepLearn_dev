
from abc import ABC, abstractmethod

from matdeeplearn.common.registry import registry
from matdeeplearn.process.utils import get_dataset

@registry.register_trainer("base")
class BaseTrainer(ABC):

    def __init__(self, task, model, dataset_config):

        #TODO: add in config for models
        self.config = {

        }
        self.dataset_config = dataset_config

        self.load()

    def load(self):
        self.load_dataset()

    def load_dataset(self):
        dataset_path = self.dataset_config['src']
        target_index = self.dataset_config.get('target_index', 0)
        self.dataset = get_dataset(dataset_path, target_index)


    @abstractmethod
    def train(self):
        """Derived classes should implement this function."""
