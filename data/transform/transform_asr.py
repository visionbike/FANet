from typing import Callable, Any
import random
import numpy as np
import torch

__all__ = [
    "ComposeAsr",
    "ToTensorAsr"
]


class ComposeAsr:
    """
    Compose class of transform functions for Custom ASR Dataset.
    """

    def __init__(self, transforms: list[Callable[..., Any]]):
        """
        :param transforms: the list of transform functions.
        """
        self.transforms = transforms

    def __call__(self, x: np.ndarray, y: Any = None) -> Any:
        """
        :param x: the input signal with shape (L, C).
        :param y: the input label.
        :return: the transformed signal and its label.
        """
        for t in self.transforms:
            x, y = t(x, y)
        return x, y


class ToTensorAsr:
    """
    ToTensor class for Custom ASR dataset.
    """

    def __call__(self, x: np.ndarray, y: Any = None) -> Any:
        """
        :param x: the input signal with shape (L, C).
        :param y: the input label.
        :return: the transformed signal and its label.
        """
        x = torch.from_numpy(x).float()
        y = torch.tensor(y).long() if y is not None else y
        return x, y
