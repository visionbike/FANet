import sys
import gc
from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from data.preprocessing import *

__all__ = ["PreprocessorAsr"]


class AsrID(Enum):
    RELAX = 0
    MOVEMENT = 1
    GRASPING = 2


class CustomAsr:
    NUM_SUBJECTS = 6
    NUM_DAYS = 1
    NUM_EXERCISES = len(AsrID) - 1
    BASE_LABEL_IDS = {
        AsrID.RELAX.value: 0,
        AsrID.MOVEMENT.value: 1,
        AsrID.GRASPING.value: 7,
    }


class PreprocessorAsr(PreprocessorBase):
    """
    Custom ASR dataset preprocessor.
    """
    def __init__(self,
                 data_root: str,
                 step_size: int = 13,
                 window_size: int = 130,
                 dc: bool = False,
                 rectif: bool = False,
                 lowpass: bool = False,
                 highpass: bool = False,
                 minmax: bool = False,
                 first_appearance: bool = True,
                 quantize: bool = False,
                 use_relax_label: bool = True,
                 exercise: int = -1):
        """
        :param data_root: the raw data root.
        :param step_size: step size for rolling windows. Default: 5.
        :param window_size: kernel size for rolling windows. Default: 52.
        :param dc: whether to use DC filter. Default: False.
        :param rectif: whether to use data rectifying. Default: False.
        :param lowpass: whether to use lowpass filter. Default: False.
        :param highpass: whether to use highpass filter. Default: False.
        :param minmax: whether to use minmax normalization. Default: False.
        :param first_appearance: whether to get first appearance label, otherwise, to get major appearance label in the windows. Default: True.
        :param quantize: whether to quantize data. Default: False.
        :param use_relax_label: whether to use the "Relax" label. Default: True.
        :param exercise: the exercise to preprocess, all exercised are used if exercise = -1. Default: -1.
        """
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.step_size = step_size
        self.window_size = window_size
        self.dc = dc
        self.rectify = rectif
        self.lowpass = lowpass
        self.highpass = highpass
        self.minmax = minmax
        self.first_appearance = first_appearance
        self.quantize = quantize
        self.use_relax_label = use_relax_label
        self.exercise = exercise
        self.ex_start = 1 if self.exercise == -1 else self.exercise
        self.ex_end = CustomAsr.NUM_EXERCISES if self.exercise == -1 else self.exercise
        super().__init__()
        self.emgs, self.lbls, self.reps, self.subs = None, None, None, None

    def _load_data_from_file(self, path: str, ex: int) -> tuple[np.ndarray, ...]:
        """
        Load MATLAB file and remap label by exercises.

        :param path: MATLAB file path.
        :param ex: the exercise index.
        :return: the loaded data arrays.
        """
        data = pd.read_csv(path)
        emgs = data[["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]].values
        # repetition labeled by a machine (more accurate labels, this is what we will use to split the data by)
        reps = data["repetition"].values[:, np.newaxis]
        # machine class exercises
        lbls = data["label"].values[:, np.newaxis]
        # remap the labels to (0, ..., 8)
        if self.exercise == -1:
            lbls = (lbls > 0).astype("int") * (CustomAsr.BASE_LABEL_IDS[ex] + lbls - 1)
        # release memory
        del data
        gc.collect()
        return emgs, lbls, reps

    def _load_data_by_subject(self, sub: int) -> tuple[list, ...]:
        """
        Load data by subject.

        :param sub: the subject index.
        :return: the loaded data arrays.
        """
        emgs, lbls, reps, subs = [], [], [], []
        for ex in tqdm(range(self.ex_start, self.ex_end + 1), file=sys.stdout):
            path = str(self.data_root / f"s{sub}" / f"S{sub}_A{ex}_D1.csv")
            data = self._load_data_from_file(path, ex)
            emgs += [data[0]]
            lbls += [data[1]]
            reps += [data[2]]
            subs += [np.full(data[2].shape, sub)]
        emgs = [np.concatenate(emgs, axis=0)]
        lbls = [np.concatenate(lbls, axis=0)]
        reps = [np.concatenate(reps, axis=0)]
        subs = [np.concatenate(subs, axis=0)]
        return emgs, lbls, reps, subs

    def load_data(self) -> None:
        """
        Load data function.
        """
        self.emgs, self.lbls, self.reps, self.subs = [], [], [], []
        for sub in range(1, CustomAsr.NUM_SUBJECTS + 1):
            print(f"Loading subject #{sub}...")
            data = self._load_data_by_subject(sub)
            self.emgs += data[0]
            self.lbls += data[1]
            self.reps += data[2]
            self.subs += data[3]
        print("Done!")

    def process_data(self, multiproc: bool = True) -> None:
        """
        Data preprocessing function

        :param multiproc: whether to use multiprocessing to process data, otherwise, to use multithreading.
        """
        if self.dc:
            print("# DC filtering...")
            self.emgs = filter_DC(self.emgs, multiproc)
        if self.highpass:
            print("# Noise reducing by high-pass filtering...")
            self.emgs = filter_highpass_butterworth(self.emgs, 20.0, 500, 3, True, multiproc)
        if self.rectify:
            print("# Rectifying ...")
            self.emgs = rectify(self.emgs, multiproc)
        if self.lowpass:
            print("# Smoothing by low-pass filtering...")
            self.emgs = filter_lowpass_butterworth(self.emgs, 20.0, 500, 1, True, multiproc)
        if self.minmax:
            print("# Min-max normalization...")
            self.emgs = norm_minmax(self.emgs, None, None, multiproc)

        print("# Rolling data windows...")
        self.emgs = roll_window(self.emgs, self.step_size, self.window_size, multiproc)
        self.lbls = roll_window(self.lbls, self.step_size, self.window_size, multiproc)
        self.reps = roll_window(self.reps, self.step_size, self.window_size, multiproc)
        self.subs = roll_window(self.subs, self.step_size, self.window_size, multiproc)
        # convert to np.ndarray
        self.emgs = np.concatenate(self.emgs, axis=0)
        self.lbls = np.concatenate(self.lbls, axis=0)
        self.reps = np.concatenate(self.reps, axis=0)
        self.subs = np.concatenate(self.subs, axis=0)
        # reshape the data in a proper axis order
        self.emgs = np.moveaxis(self.emgs, 2, 1)
        self.lbls = np.moveaxis(self.lbls, 2, 1)[..., -1]
        self.reps = np.moveaxis(self.reps, 2, 1)[..., -1]
        self.subs = np.moveaxis(self.subs, 2, 1)[..., -1]

        print("# Removing windows with multiple repetitions...")
        # split by repetition without any data leaks by drop any windows having more than one repetition
        no_leaks = np.array([i for i in range(self.reps.shape[0]) if np.unique(self.reps[i]).shape[0] == 1])
        self.emgs = self.emgs[no_leaks, :, :]
        self.lbls = self.lbls[no_leaks, :]
        self.reps = self.reps[no_leaks, :]
        self.subs = self.subs[no_leaks, :]
        # release memory
        del no_leaks
        gc.collect()

        if self.first_appearance:
            print("# Replacing labels by the first appearance label...")
            self.lbls = np.asarray(get_first_appearance(list(self.lbls), multiproc))
            self.reps = np.asarray(get_first_appearance(list(self.reps), multiproc))
            self.subs = np.asarray(get_first_appearance(list(self.subs), multiproc))
        else:
            print("# Replacing labels by the major appearance label...")
            self.lbls = np.asarray(get_major_appearance(list(self.lbls), multiproc))
            self.reps = np.asarray(get_major_appearance(list(self.reps), multiproc))
            self.subs = np.asarray(get_major_appearance(list(self.subs), multiproc))

        if self.quantize:
            print("# Quantifying to float16...")
            self.emgs = self.emgs.astype(np.float16)

        if not self.use_relax_label:
            print("# Discarding 'relax' label...")
            self.emgs = self.emgs[np.where(self.lbls != 0)[0]]
            self.reps = self.reps[np.where(self.lbls != 0)[0]]
            self.subs = self.subs[np.where(self.lbls != 0)[0]]
            self.lbls = self.lbls[np.where(self.lbls != 0)[0]]
            self.lbls -= 1
        print("Done!")

    def split_data(self, mode: str, save_path: str, valtest_reps: list = None) -> None:
        """
         Split data with different evaluation modes.
        :param mode: validation mode:
            * "trainvaltest": splitting data for train/val/test sets.
            * "session": splitting data for inter-session cross validation.
            * "subject": splitting data for inter-subject cross validation.
            * subject-adaptive-transfer: splitting for subject-adaptive transferring.
        :param save_path: the path for processed data.
        :param valtest_reps: the repetition indices for validation and test sets. Default: None.
        """
        path_save = Path(save_path) / mode
        path_save.mkdir(parents=True, exist_ok=True)
        print("# Splitting mode = {mode}...")
        if mode == "trainvaltest":
            reps_unique = np.unique(self.reps)
            # split train dataset
            reps = reps_unique[np.where(np.isin(reps_unique, valtest_reps, invert=True))]
            idxs = np.where(np.isin(self.reps, reps))
            data_train = dict(emg=self.emgs[idxs].copy(), lbl=self.lbls[idxs].copy())
            # split valid dataset
            reps = valtest_reps[:-2]
            idxs = np.where(np.isin(self.reps, reps))
            data_val = dict(emg=self.emgs[idxs].copy(), lbl=self.lbls[idxs].copy())
            # split test dataset
            reps = valtest_reps[-2:]
            idxs = np.where(np.isin(self.reps, reps))
            data_test = dict(emg=self.emgs[idxs].copy(), lbl=self.lbls[idxs].copy())
            print(f"train data: shape=(emg: {data_train['emg'].shape}, imu: {data_train['imu'].shape}), train target: shape={data_train['lbl'].shape}")
            print(f"test data: shape=(emg: {data_test['emg'].shape}, imu: {data_test['imu'].shape}), test target: shape={data_test['lbl'].shape}")
            print(f"val data: shape=(emg: {data_val['emg'].shape}, imu: {data_val['imu'].shape}), val target: shape={data_val['lbl'].shape}")
            print("# Saving processed data...")
            np.savez(str(path_save / "train.npz"), **data_train)
            np.savez(str(path_save / "test.npz"), **data_test)
            np.savez(str(path_save / "val.npz"), **data_val)
            # release memory
            del data_train, data_val, data_test
            gc.collect()
        elif mode == "session":
            reps_unique = np.unique(self.reps)
            for rep in reps_unique:
                print(f"Repetition #{rep}...")
                # get repetitions for train/val/test sets...
                reps_trainval = reps_unique[np.where(np.isin(reps_unique, rep, invert=True))]
                reps_val = reps_trainval[0 if rep == len(reps_unique) else rep]
                reps_train = np.setdiff1d(reps_trainval, reps_val)
                reps_test = rep
                # split train set
                idxs = np.where(np.isin(self.reps, reps_train))
                data_train = dict(emg=self.emgs[idxs].copy(), lbl=self.lbls[idxs].copy())
                # split val set
                idxs = np.where(np.isin(self.reps, reps_val))
                data_val = dict(emg=self.emgs[idxs].copy(), lbl=self.lbls[idxs].copy())
                # split test set
                idxs = np.where(np.isin(self.reps, reps_test))
                data_test = dict(emg=self.emgs[idxs].copy(), lbl=self.lbls[idxs].copy())
                print(f"train data: shape=(emg: {data_train['emg'].shape}, imu: {data_train['imu'].shape}), train target: shape={data_train['lbl'].shape}")
                print(f"test data: shape=(emg: {data_test['emg'].shape}, imu: {data_test['imu'].shape}), test target: shape={data_test['lbl'].shape}")
                print(f"val data: shape=(emg: {data_val['emg'].shape}, imu: {data_val['imu'].shape}), val target: shape={data_val['lbl'].shape}")
                print("# Saving processed data...")
                np.savez(str(path_save / f"train_fold{rep + 1}.npz"), **data_train)
                np.savez(str(path_save / f"test_fold{rep + 1}.npz"), **data_test)
                np.savez(str(path_save / f"val_fold{rep + 1}.npz"), **data_val)
                # release memory
                del data_train, data_val, data_test
                gc.collect()
        elif mode == "subject":
            reps_unique = np.unique(self.reps)
            subs_unique = np.unique(self.subs)
            for sub in subs_unique:
                print(f"Subject #{sub}...")
                # get subjects for train/val/test sets...
                subs_trainval = subs_unique[np.where(np.isin(subs_unique, sub, invert=True))]
                subs_val = subs_trainval[0 if sub == len(subs_unique) else (sub - 1)]
                subs_train = np.setdiff1d(subs_trainval, subs_val)
                subs_test = sub
                # split train set
                sub_idxs = np.where(np.isin(self.subs, subs_train))
                idxs = np.intersect1d(sub_idxs, np.where(np.isin(self.reps, reps_unique)))
                data_train = dict(emg=self.emgs[idxs].copy(), lbl=self.lbls[idxs].copy())
                # split val set
                sub_idxs = np.where(np.isin(self.subs, subs_val))
                idxs = np.intersect1d(sub_idxs, np.where(np.isin(self.reps, reps_unique)))
                data_val = dict(emg=self.emgs[idxs].copy(), lbl=self.lbls[idxs].copy())
                # split test set
                sub_idxs = np.where(np.isin(self.subs, subs_test))
                idxs = np.intersect1d(sub_idxs, np.where(np.isin(self.reps, reps_unique)))
                data_test = dict(emg=self.emgs[idxs].copy(), lbl=self.lbls[idxs].copy())
                print(f"train data: shape=(emg: {data_train['emg'].shape}, imu: {data_train['imu'].shape}), train target: shape={data_train['lbl'].shape}")
                print(f"test data: shape=(emg: {data_test['emg'].shape}, imu: {data_test['imu'].shape}), test target: shape={data_test['lbl'].shape}")
                print(f"val data: shape=(emg: {data_val['emg'].shape}, imu: {data_val['imu'].shape}), val target: shape={data_val['lbl'].shape}")
                print("# Saving processed data...")
                np.savez(str(path_save / f"train_fold{sub}.npz"), **data_train)
                np.savez(str(path_save / f"test_fold{sub}.npz"), **data_test)
                np.savez(str(path_save / f"val_fold{sub}.npz"), **data_val)
                # release memory
                del data_train, data_val, data_test
                gc.collect()
        elif mode == "subject-adaptive-transfer":
            reps_unique = np.unique(self.reps)
            subs_unique = np.unique(self.subs)
            for sub in subs_unique:
                print(f"Subject #{sub}...")
                ### pretraining data
                sub_idxs = np.where(np.isin(self.subs, sub))
                # split train set
                pretrain_idxs = np.intersect1d(sub_idxs, np.where(np.isin(self.reps, valtest_reps, invert=True)))
                data_pretrain_train = dict(emg=self.emgs[pretrain_idxs].copy(), lbl=self.lbls[pretrain_idxs].copy())
                # split val set
                pretrain_idxs = np.intersect1d(sub_idxs, np.where(np.isin(self.reps, valtest_reps[:-1])))
                data_pretrain_val = dict(emg=self.emgs[pretrain_idxs].copy(), lbl=self.lbls[pretrain_idxs].copy())
                # split test set
                pretrain_idxs = np.intersect1d(sub_idxs, np.where(np.isin(self.reps, valtest_reps[-1])))
                data_pretrain_test = dict(emg=self.emgs[pretrain_idxs].copy(), lbl=self.lbls[pretrain_idxs].copy())
                ### transferring data
                transfer_idxs = np.intersect1d(sub_idxs, np.where(np.isin(self.reps, reps_unique[:3])))
                data_transfer_train = dict(emg=self.emgs[transfer_idxs].copy(), lbl=self.lbls[transfer_idxs].copy())
                transfer_idxs = np.intersect1d(sub_idxs, np.where(np.isin(self.reps, reps_unique[3:5])))
                data_transfer_val = dict(emg=self.emgs[transfer_idxs].copy(), lbl=self.lbls[transfer_idxs].copy())
                transfer_idxs = np.intersect1d(sub_idxs, np.where(np.isin(self.reps, reps_unique[5:])))
                data_transfer_test = dict(emg=self.emgs[transfer_idxs].copy(), lbl=self.lbls[transfer_idxs].copy())
                print(f"pretrain train data: shape=(emg: {data_pretrain_train['emg'].shape}, imu: {data_pretrain_train['imu'].shape}), pretrain train target: shape={data_pretrain_train['lbl'].shape}")
                print(f"pretrain test data: shape=(emg: {data_pretrain_test['emg'].shape}, imu: {data_pretrain_test['imu'].shape}), pretrain test target: shape={data_pretrain_test['lbl'].shape}")
                print(f"pretrain val data: shape=(emg: {data_pretrain_val['emg'].shape}, imu: {data_pretrain_val['imu'].shape}), pretrain val target: shape={data_pretrain_val['lbl'].shape}")
                print(f"transfer train data: shape=(emg: {data_transfer_train['emg'].shape}, imu: {data_transfer_train['imu'].shape}), transfer train target: shape={data_transfer_train['lbl'].shape}")
                print(f"transfer test data: shape=(emg: {data_transfer_test['emg'].shape}, imu: {data_transfer_test['imu'].shape}), transfer test target: shape={data_transfer_test['lbl'].shape}")
                print(f"transfer val data: shape=(emg: {data_transfer_val['emg'].shape}, imu: {data_transfer_val['imu'].shape}), transfer val target: shape={data_transfer_val['lbl'].shape}")
                print("# Saving processed data...")
                np.savez(str(path_save / f"pretrain_train_fold{sub}.npz"), **data_pretrain_train)
                np.savez(str(path_save / f"pretrain_test_fold{sub}.npz"), **data_pretrain_test)
                np.savez(str(path_save / f"pretrain_val_fold{sub}.npz"), **data_pretrain_val)
                np.savez(str(path_save / f"transfer_train_fold{sub}.npz"), **data_transfer_train)
                np.savez(str(path_save / f"transfer_test_fold{sub}.npz"), **data_transfer_test)
                np.savez(str(path_save / f"transfer_val_fold{sub}.npz"), **data_transfer_val)
                # release memory
                del data_pretrain_train, data_pretrain_val, data_pretrain_test, data_transfer_train, data_transfer_val, data_transfer_test
                gc.collect()
        print("Done!")
