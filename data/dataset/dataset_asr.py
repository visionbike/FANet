import gc
import numpy as np
import torch
from torch.utils.data import Dataset
from data.transform import ComposeAsr, ToTensorAsr

__all__ = ["DatasetAsr"]


class DatasetAsr(Dataset):
    """
    Custom ASR Dataset.
    """

    def __init__(self,
                 data: np.ndarray,
                 use_relax_label: bool = True) -> None:
        """
        :param data: the Nina DB5 data, include signal and its label.
        :param use_relax_label: whether to use the "relax" label.
        """
        super().__init__()
        # load data
        self.data = data['emg'].copy()
        self.lbls = data['lbl'].copy()
        # release memory
        del data
        gc.collect()
        # get transformation
        transforms = []
        transforms += [ToTensorAsr()]
        self.transforms = ComposeAsr(transforms)

    def __len__(self) -> int:
        return self.lbls.shape[0]

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        """
        :param idx: the index of the data.
        :return: the pytorch's tensor representation of the input and its label.
        """
        # shape: (L, C)
        x = self.data[idx]
        y = self.lbls[idx]
        x, y = self.transforms(x, y)
        return x, y
