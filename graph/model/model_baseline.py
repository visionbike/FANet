import torch
import lightning as l
from graph.network import *
from graph.network.init import *
from graph.optimizer import *
from graph.loss import *
from graph.lr_scheduler import *
from graph.metric import *
from utils import *

__all__ = ["ModelBaseline"]


class ModelBaseline(l.LightningModule):
    """
    Model Baseline Lightning module.
    """

    def __init__(self,
                 network_kwargs: dict,
                 criterion_kwargs: dict,
                 optimizer_kwargs: dict,
                 scheduler_kwargs: dict,
                 metric_kwargs: dict,
                 logger: str = "neptune") -> None:
        """
        :param network_kwargs: the network configuration.
        :param criterion_kwargs: the criterion configuration.
        :param optimizer_kwargs: the optimizer configuration.
        :param scheduler_kwargs: the LR scheduler configuration.
        :param metric_kwargs: the metric configuration.
        :param logger: logger name. Default: "neptune".
        """
        super().__init__()

        self.net = get_network(**network_kwargs)
        self.criterion = get_loss(**criterion_kwargs)
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs

        self.metric_train = get_metrics(prefix="train/", **metric_kwargs)
        self.metric_val = get_metrics(prefix="val/", **metric_kwargs)
        self.metric_test = get_metrics(prefix="test/", cm=True, **metric_kwargs)
        self.logger_ = logger
        self.best_acc, self.best_bacc, self.best_mcc, self.best_f1 = 0.0, 0.0, 0.0, 0.0

    def reset_parameters(self) -> None:
        """
        Reset the network weights.
        """
        self.net.apply(init_weights)

    def freeze(self, proj: bool = True, att: bool = True, head: bool = False) -> None:
        """
        Freeze the module's weights for transfer learning.

        :param proj: whether to freeze projection layer's weights. Default: True.
        :param att: whether to freeze projection layer's weights. Default: True.
        :param head: whether to freeze projection layer's weights. Default: False.
        """
        if proj:
            self.net.proj.requires_grad_(False)
        if att:
            self.net.att.requires_grad_(False)
        if head:
            self.net.head.requires_grad_(False)

    def load_net_state_dict(self, path: str) -> None:
        """
        Load the network weights from file.

        :param path: the network's weight file.
        """
        self.net.load_state_dict(torch.load(path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: the input tensor in shape of (N, L, C).
        :return: the output tensor in shape of (N, L, C).
        """
        return self.net(x)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.net.parameters(), **self.optimizer_kwargs)
        scheduler = get_lr_scheduler(optimizer, **self.scheduler_kwargs)
        if scheduler is not None:
            lr_scheduler_config = dict(
                scheduler=scheduler, # required scheduler instance
                interval="epoch",    # update it after optimizer update ("step", "epoch")
                frequency=1,         # how many epochs/steps should pass to update learning rate
                monitor="val/loss",  # metric to monitor for schedulers
            )
            return [optimizer], [lr_scheduler_config]
        return [optimizer]

    def training_step(self, batch: tuple, batch_idx: int) -> dict:
        x, y = batch
        z = self.forward(x)
        loss = self.criterion(z, y)
        self.metric_train.update(z, y)   # update metric for on_epoch
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return dict(loss=loss)

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.metric_train, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        x, y = batch
        z = self.forward(x)
        loss = self.criterion(z, y)
        self.metric_val.update(z, y)  # update metric for on_epoch
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.metric_val, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        x, y = batch
        z = self.forward(x)
        self.metric_test.update(z, y)   # update metric for on_epoch

    def on_test_epoch_end(self) -> None:
        # turn confusion matrix into a figure (a tensor cannot be logged as a scalar)
        metric_test_results = self.metric_test.compute()
        # log figure
        confmat = metric_test_results.pop("test/confusion_matrix").detach().cpu().numpy()
        fig_ = visualize_confusion_matrix(confmat)
        if self.logger_ == "neptune":
            from neptune.types import File
            for acc in confmat.diagonal():
                self.logger.experiment["test/per_class_accuracy"].append(acc)
            self.logger.experiment["test/confusion_matrix"].upload(File.as_image(fig_))
        elif self.logger_ == "tensorboard":
            self.logger.experiment.add_figure("test/confusion_matrix", fig_, global_step=self.current_epoch)
        self.log_dict(metric_test_results, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.best_acc = metric_test_results["test/accuracy"]
        self.best_bacc = metric_test_results["test/balanced_accuracy"]
        self.best_mcc = metric_test_results["test/mathews_corr_coef"]
        self.best_f1 = metric_test_results["test/f1_score"]
