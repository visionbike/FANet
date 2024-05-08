import gc
from typing import Callable
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Dataset
import lightning as l

__all__ = ["DataloaderBase"]


class DataloaderBase(l.LightningDataModule):
    """
    The LightningData of base dataloader.
    """

    def __init__(self,  dataset: Callable | Dataset, **kwargs):
        """
        :param dataset: the dataset callable.
        :param kwargs: arguments for dataloader.
        """
        super().__init__()
        data_root = Path(kwargs["path"])
        if not data_root.exists():
            raise FileNotFoundError(f"No such file or directory: {data_root}.")
        self.data_root = data_root

        self.dataset = dataset
        self.kwargs_dataset = kwargs["DatasetConfig"]
        self.kwargs_dataloader = {
            "num_classes": kwargs["num_classes"],
            "batch_size": kwargs["batch_size"],
            "num_workers": kwargs["num_workers"],
            "persistent_workers": True,
            "pin_memory": True,
            "shuffle": True
        }
        self.dataset_train, self.dataset_val, self.dataset_test = None, None, None

    def setup(self, stage: str = None):
        """
        Setup for the dataloader.

        :param stage: setup dataloader for train/val/test sets. Default: None.
        """
        if stage in [None, "fit"]:
            data_train = np.load(str(self.data_root / "train.npz"), allow_pickle=True)
            data_val = np.load(str(self.data_root / "val.npz"), allow_pickle=True)
            self.dataset_train = self.dataset(data_train, **self.kwargs_dataset)
            self.dataset_val = self.dataset(data_val, **self.kwargs_dataset)
            # release memory
            del data_train, data_val
            gc.collect()
        elif stage == "test":
            data_test = np.load(str(self.data_root / "test.npz"), allow_pickle=True)
            self.dataset_test = self.dataset(data_test, **self.kwargs_dataset)
            # release memory
            del data_test
            gc.collect()
        else:
            raise ValueError(f"Invalid stage name, got {stage}.")

    def train_dataloader(self):
        self.kwargs_dataloader["shuffle"] = True
        return DataLoader(self.dataset_train, **self.kwargs_dataloader)

    def val_dataloader(self):
        self.kwargs_dataloader["shuffle"] = False
        return DataLoader(self.dataset_val, **self.kwargs_dataloader)

    def test_dataloader(self):
        self.kwargs_dataloader["shuffle"] = False
        return DataLoader(self.dataset_test, **self.kwargs_dataloader)
