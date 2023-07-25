from matdeeplearn.common.trainer_context import new_trainer_context
from matdeeplearn.trainers.property_trainer import PropertyTrainer

from numpy import floating, integer

def trainer_context(config, train: bool):
    with new_trainer_context(args=None, config=config) as ctx:
        task = ctx.task
        trainer = ctx.trainer
        task.setup(trainer)
        if train:
            task.run()
    return trainer

def trainer_property(config, train: bool):
    trainer = PropertyTrainer.from_config(config)
    if train:
        trainer.train()
    return trainer

def assert_valid_predictions(trainer, load: str):
    try:
        out = trainer.predict(loader=trainer.data_loader[load], split="predict", write_output=False)
        assert isinstance(out["predict"][0][0], (floating, float, integer, int))
        assert isinstance(out["ids"][0][0], str)    
        if load != "predict_loader":
            assert isinstance(out["target"][0][0], (floating, float, integer, int))
    except:
        assert False   
        