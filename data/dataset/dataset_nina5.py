import gc
import numpy as np
import torch
from torch.utils.data import Dataset
from data.transform import ComposeNina5, ToTensorNina5

__all__ = ["DatasetNina5"]


class DatasetNina5(Dataset):
    """
    NinaPro DB5 Dataset.
    """

    def __init__(self, data: np.ndarray, use_relax_label: bool = True):
        """
        :param data: the Nina DB5 data, include signal and its label.
        :param use_relax_label: whether to use the "relax" label. Default: True
        """
        super().__init__()
        # load data
        if data["imu"].size == 0:
            self.data = data["emg"].copy()
        else:
            self.data = np.concatenate([data["emg"].copy(), data["imu"].copy()], axis=-1)
        self.lbls = data["lbl"].copy()
        # release memory
        del data
        gc.collect()
        # get transformation
        transforms = []
        transforms += [ToTensorNina5()]
        self.transforms = ComposeNina5(transforms)

    def __len__(self) -> int:
        return self.lbls.shape[0]

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        """
        :param idx: the index of the data.
        :return: the pytorch's tensor representation of the input and its label
        """
        # shape: (L, C)
        x = self.data[idx]
        y = self.lbls[idx]
        x, y = self.transforms(x, y)
        return x, y
