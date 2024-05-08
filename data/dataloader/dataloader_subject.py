import gc
from typing import Callable
import numpy as np
from torch.utils.data import Dataset
from .dataloader_base import *

__all__ = ["DataloaderSubject"]


class DataloaderSubject(DataloaderBase):
    """
    The LightningData of dataloader for inter-session cross validation.
    """

    def __init__(self, dataset: Callable | Dataset, **kwargs):
        """
        :param dataset: the dataset callable.
        :param kwargs: arguments for dataloader.
        """
        super().__init__(dataset, **kwargs)

    def setup(self, stage: str = None, k: int = 1):
        """
        Setup for the dataloader.

        :param stage: setup dataloader for train/val/test sets. Default: None.
        :param k: the session index. Default: 1.
        """
        if stage in [None, "fit"]:
            data_train = np.load(str(self.data_root / f"train_fold{k}.npz"), allow_pickle=True)
            data_val = np.load(str(self.data_root / f"val_fold{k}.npz"), allow_pickle=True)
            self.dataset_train = self.dataset(data_train, **self.kwargs_dataset)
            self.dataset_val = self.dataset(data_val, **self.kwargs_dataset)
            # release memory
            del data_train, data_val
            gc.collect()
        elif stage == "test":
            data_test = np.load(str(self.data_root / f"test{k}.npz"), allow_pickle=True)
            self.dataset_test = self.dataset(data_test, **self.kwargs_dataset)
            # release memory
            del data_test
            gc.collect()
        else:
            raise ValueError(f"Invalid stage name, got {stage}.")
