from matdeeplearn.trainers.base_trainer import BaseTrainer
from matdeeplearn.common.registry import registry

@registry.register_trainer("property")
class PropertyTrainer(BaseTrainer):
    def __init__(self, task, model, dataset_config):
        super().__init__(task, model, dataset_config)

    def train(self):
        #TODO add in training functionality
        pass
