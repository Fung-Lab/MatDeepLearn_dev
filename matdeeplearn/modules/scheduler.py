class LRScheduler:
    def __init__(self, optimizer, scheduler, scheduler_args):
        self.optimizer = optimizer
        self.scheduler_type = scheduler
