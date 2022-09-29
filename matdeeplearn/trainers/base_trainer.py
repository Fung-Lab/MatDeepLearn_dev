
from abc import ABC, abstractmethod

from matdeeplearn.common.registry import registry

@registry.register_trainer("base")
class BaseTrainer(ABC):

    def __init__(self):

        #TODO: add in config for models
        self.config = {

        }

    def load(self):
        self.load_dataset()

    def load_dataset(self):
        # TODO: call get_dataset() from process()
        pass


    @abstractmethod
    def train(self):
        """Derived classes should implement this function."""
