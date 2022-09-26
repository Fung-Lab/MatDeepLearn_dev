
from abc import ABC, abstractmethod

from matdeeplearn.common.registry import registry

@registry.register_trainer("base")
class BaseTrainer(ABC):

    def __init__(self):

        #TODO: add in config for models
        self.config = {

        }


    @abstractmethod
    def train(self):
        """Derived classes should implement this function."""
