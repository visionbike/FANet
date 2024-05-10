import torch
from torch import nn
from torch.nn import functional as fn

__all__ = ["FocalLoss"]


class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss from the paper in multiclass classification.
    The losses are averaged across observations for each mini-batch. This focal loss supports label smoothing.

    Formula:
    loss = -alpha * (1 - p)^gamma) * log(p)
    y_ls = (1 - smooth) * y_hot + smooth / num_classes

    References:
    * Focal Loss for Dense Object Detection (https://arxiv.org/abs/1708.02002)
    * https://github.com/pytorch/vision/issues/3250
    * https://github.com/JohannesLiu/Deep-Learning-Loss-Function-Collection-for-Imbalanced-Data/blob/main/losses/FocalLoss.py
    * https://github.com/ashawkey/FocalLoss.pytorch/blob/master/focalloss.py
    * https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
    * https://github.com/Kageshimasu/focal-loss-with-smoothing/blob/main/focal_loss_with_smoothing.py
    """

    def __init__(self,
                 num_classes: int,
                 alpha: torch.Tensor | list = None,
                 gamma: float = 2,
                 smoothing: float = 0.1,
                 reduction: str = "mean"):
        """
        :param num_classes: the number of classes.
        :param alpha: specifying per-example weight for balanced cross entropy. Default. None.
        :param gamma: scalar modulating loss from hard and easy example. Default: 2.0.
        :param smoothing: label smoothing value for slowdown the over-fitting. Default: 0.1.
        :param reduction: averaging method: "mean", "sum" or "none". Defaults to "mean".
        """
        if num_classes <= 1:
            raise ValueError(f"Invalid num_classes value, but got {num_classes}")
        if gamma < 0:
            raise ValueError(f"Invalid gamma value, but got {gamma}.")
        if reduction not in ["mean", "sum"]:
            raise ValueError(f"Invalid reduction value, but got {reduction}.")
        super().__init__()

        if alpha is None:
            alpha = torch.ones((num_classes,), dtype=torch.float32, requires_grad=False)
        elif isinstance(alpha, list):
            alpha = torch.tensor(alpha, dtype=torch.float32, requires_grad=False)

        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.eps = 1e-10
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        :param logits: the logit tensor in shape of (N, num_classes).
        :param labels: the label tensor in shape of (N,).
        :return: the normalized focal loss.
        """
        # compute onehot labels
        labels_hot = fn.one_hot(labels, num_classes=self.num_classes)
        # add label smoothing into prediction value
        preds = (1 - self.smoothing) * torch.softmax(logits, dim=-1) + self.smoothing / self.num_classes
        # clip the prediction value
        preds = torch.clip(preds, self.eps, 1. - self.eps)
        # compute cross-entropy
        ce = - labels_hot * torch.log(preds)
        # calculate weight that consists of  modulating factor and weighting factor
        # get corresponding class weight value
        weight = self.alpha.to(logits.device) * labels_hot * (1 - preds) ** self.gamma
        # compute focal loss
        loss = weight * ce
        loss = loss.sum(dim=-1).mean() if self.reduction == "mean" else loss.sum(dim=-1).sum()
        return loss
