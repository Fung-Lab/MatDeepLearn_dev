from matdeeplearn.trainers.base_trainer import BaseTrainer
from matdeeplearn.common.registry import registry


@registry.register_trainer("property")
class PropertyTrainer(BaseTrainer):
    def __init__(self, model, dataset, optimizer, sampler, train_loader, loss, max_epochs):
        super().__init__(model, dataset, optimizer, sampler, train_loader, loss, max_epochs)

    def train(self):
        # Start training over epochs loop
        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)

        for epoch in range(start_epoch, self.max_epochs):
            self.train_sampler.set_epoch(epoch)
            skip_steps = self.step % len(self.train_loader)
            train_loader_iter = iter(self.train_loader)

            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch + (i + 1) / len(self.train_loader)
                self.step = epoch * len(self.train_loader) + i + 1
                self.model.train()

                # Get a batch of train data
                batch = next(train_loader_iter)

                # Compute forward, loss, backward
                out = self._forward(batch)
                loss = self._compute_loss(out, batch)
                self._backward(loss)

                # Compute metrics
                metrics = self._compute_metrics(out, batch, {})
                self.metrics = self.evaluator.update("loss", loss.item(), metrics)

                # Log metrics
                self._log_metrics()

                # Evaluate on validation set if it exists
                # TODO: could add param to eval on increments instead of every time
                if self.val_loader:
                    self.val_metrics = self.validate()
                    # save checkpoint if metric is best so far
                    # if it is best and test loader exists, then predict too

                # step scheduler, using validation error
                self._scheduler_step()

    def validate(self):
        # TODO: implement validate method
        return {}

    def predict(self):
        # TODO: implement predict method
        return {}

    def _forward(self, batch_data):
        output = self.model(batch_data)
        return output

    def _compute_loss(self, out, batch_data):
        loss = self.loss_fn(out, batch_data.y.to(self.rank))
        return loss

    def _backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _compute_metrics(self, out, batch_data, metrics):
        # TODO: finish this method
        metrics = self.evaluator.eval()
        return metrics

    def _log_metrics(self):
        pass

    def _scheduler_step(self):
        if self.scheduler.scheduler_type == "ReduceLROnPlateau":
            self.scheduler.step(
                metrics=self.val_metrics["metric"]
            )
        else:
            self.scheduler.step()
