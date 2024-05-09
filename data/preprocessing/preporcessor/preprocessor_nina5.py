import sys
import gc
from enum import Enum
from pathlib import Path
import numpy as np
import scipy.io as scio
from tqdm import tqdm
from .preprocessor_base import *
from data.preprocessing import *

__all__ = [""]


class Nina5ID(Enum):
    RELAX = 0
    A = 1
    B = 2
    C = 3


class Nina5:
    NUM_SUBJECTS = 10
    NUM_DAYS = 1
    NUM_EXERCISES = len(Nina5ID) - 1
    BASE_LABEL_IDS = {
        Nina5ID.RELAX.value: 0,
        Nina5ID.A.value: 1,
        Nina5ID.B.value: 13,
        Nina5ID.C.value: 30
    }


class PreprocessorNina5(PreprocessorBase):
    """
    NinaPro DB5 preprocessor.
    """
    def __init__(self,
                 data_root: str,
                 step_size: int = 5,
                 window_size: int = 52,
                 imu: bool = False,
                 dc: bool = False,
                 rectif: bool = False,
                 lowpass: bool = False,
                 highpass: bool = False,
                 minmax: bool = False,
                 first_appearance: bool = True,
                 quantize: bool = False,
                 use_relax_label: bool = True,
                 exercise: int = -1):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.step_size = step_size
        self.window_size = window_size
        self.imu = imu
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
        self.ex_end = Nina5.NUM_EXERCISES if self.exercise == -1 else self.exercise
        self.emgs, self.imus, self.lbls, self.reps, self.subs = None, None, None, None, None

    def _load_data_from_file(self, path: str, ex: int) -> tuple[np.ndarray, ...]:
        """
        Load MATLAB file and remap label by exercises.

        :param path: MATLAB file path.
        :param ex: the exercise index.
        :return: the loaded data arrays.
        """
        data = scio.loadmat(path)
        emgs = data["emg"].copy()
        imus = data["acc"].copy() if self.imu else []
        # repetition labeled by a machine (more accurate labels, this is what we will use to split the data by)
        reps = data["rerepetition"].copy()
        # class exercises
        lbls = data["restimulus"].copy()
        # remap the labels to (0, ..., 52)
        if self.exercise == -1:
            lbls = (lbls > 0).astype("int") * (Nina5.BASE_LABEL_IDS[ex] + lbls - 1)
        # release memory
        del data
        gc.collect()
        return emgs, imus, lbls, reps

    def _load_data_by_subject(self, sub: int) -> tuple[list, ...]:
        """
        Load data by subject.

        :param sub: the subject index.
        :return: the loaded data arrays.
        """
        emgs, imus, lbls, reps, subs = [], [], [], [], []
        for ex in tqdm(range(self.ex_start, self.ex_end + 1), file=sys.stdout):
            path = self.data_root / f"s{sub}" / f"S{sub}_E{ex}_A1.mat"
            data = self._load_data_from_file(path, ex)
            emgs += [data[0]]
            imus += [data[1]] if self.imu else data[1]
            lbls += [data[2]]
            reps += [data[3]]
            subs += [np.full(data[3].shape, sub)]
        emgs = [np.concatenate(emgs, axis=0)]
        imus = [np.concatenate(imus, axis=0)] if self.imu else imus
        lbls = [np.concatenate(lbls, axis=0)]
        reps = [np.concatenate(reps, axis=0)]
        subs = [np.concatenate(subs, axis=0)]
        return emgs, imus, lbls, reps, subs

    def load_data(self) -> None:
        """
        Load data function.
        """
        self.emgs, self.imus, self.lbls, self.reps, self.subs = [], [], [], [], []
        for sub in range(1, Nina5.NUM_SUBJECTS + 1):
            print(f"Loading subject #{sub}...")
            data = self._load_data_by_subject(sub)
            self.emgs += data[0]
            self.imus += data[1]
            self.lbls += data[2]
            self.reps += data[3]
            self.subs += data[4]
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
            self.emgs = filter_highpass_butterworth(self.emgs, 1.0, 200, 4, True, multiproc)
        if self.rectify:
            print("# Rectifying ...")
            self.emgs = rectify(self.emgs, multiproc)
        if self.lowpass:
            print("# Smoothing by low-pass filtering...")
            self.emgs = filter_lowpass_butterworth(self.emgs, 2.0, 200, 4, True, multiproc)
        if self.minmax:
            print("# Min-max normalization...")
            self.emgs = norm_minmax(self.emgs, None, None, multiproc)

        print("# Rolling data windows...")
        self.emgs = roll_window(self.emgs, self.step_size, self.window_size, multiproc)
        self.imus = roll_window(self.imus, self.step_size, self.window_size, multiproc) if self.imu else np.asarray(self.imus)
        self.lbls = roll_window(self.lbls, self.step_size, self.window_size, multiproc)
        self.reps = roll_window(self.reps, self.step_size, self.window_size, multiproc)
        self.subs = roll_window(self.subs, self.step_size, self.window_size, multiproc)
        # convert to np.ndarray
        self.emgs = np.concatenate(self.emgs, axis=0)
        self.imus = np.concatenate(self.imus, axis=0)
        self.lbls = np.concatenate(self.lbls, axis=0)
        self.reps = np.concatenate(self.reps, axis=0)
        self.subs = np.concatenate(self.subs, axis=0)
        # reshape the data in a proper axis order
        self.emgs = np.moveaxis(self.emgs, 2, 1)
        self.imus = np.moveaxis(self.imus, 2, 1) if self.imu else self.imus
        self.lbls = np.moveaxis(self.lbls, 2, 1)[..., -1]
        self.reps = np.moveaxis(self.reps, 2, 1)[..., -1]
        self.subs = np.moveaxis(self.subs, 2, 1)[..., -1]

        print("# Removing windows with multiple repetitions...")
        # split by repetition without any data leaks by drop any windows having more than one repetition
        no_leaks = np.array([i for i in range(self.reps.shape[0]) if np.unique(self.reps[i]).shape[0] == 1])
        self.emgs = self.emgs[no_leaks, :, :]
        self.imus = self.imus[no_leaks, :, :] if self.imu else self.imus
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
            self.imus = self.imus.astype(np.float16) if self.imu else self.imus

        if not self.use_relax_label:
            print("# Discarding 'relax' label...")
            self.emgs = self.emgs[np.where(self.lbls != 0)[0]]
            self.imus = self.imus[np.where(self.lbls != 0)[0]] if self.imu else self.imus
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
            reps_unique = np.unique(self.reps)[1:]  # discard session 0
            # split train dataset
            reps = reps_unique[np.where(np.isin(reps_unique, valtest_reps, invert=True))]
            idxs = np.where(np.isin(self.reps, reps))
            data_train = dict(emg=self.emgs[idxs].copy(),
                              imu=self.imus[idxs].copy() if self.imu else self.imus,
                              lbl=self.lbls[idxs].copy())
            # split valid dataset
            reps = valtest_reps[:-1]
            idxs = np.where(np.isin(self.reps, reps))
            data_val = dict(emg=self.emgs[idxs].copy(),
                            imu=self.imus[idxs].copy() if self.imu else self.imus,
                            lbl=self.lbls[idxs].copy())
            # split test dataset
            reps = valtest_reps[-1]
            idxs = np.where(np.isin(self.reps, reps))
            data_test = dict(emg=self.emgs[idxs].copy(),
                             imu=self.imus[idxs].copy() if self.imu else self.imus,
                             lbl=self.lbls[idxs].copy())
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
            reps_unique = np.unique(self.reps)[1:]  # discard session 0
            for rep in reps_unique:
                print(f"Repetition #{rep}...")
                # get repetitions for train/val/test sets...
                reps_trainval = reps_unique[np.where(np.isin(reps_unique, rep, invert=True))]
                reps_val = reps_trainval[0 if rep == len(reps_unique) else (rep - 1)]
                reps_train = np.setdiff1d(reps_trainval, reps_val)
                reps_test = rep
                # split train set
                idxs = np.where(np.isin(self.reps, reps_train))
                data_train = dict(emg=self.emgs[idxs].copy(),
                                  imu=self.imus[idxs].copy() if self.imu else self.imus,
                                  lbl=self.lbls[idxs].copy())
                # split val set
                idxs = np.where(np.isin(self.reps, reps_val))
                data_val = dict(emg=self.emgs[idxs].copy(),
                                imu=self.imus[idxs].copy() if self.imu else self.imus,
                                lbl=self.lbls[idxs].copy())
                # split test set
                idxs = np.where(np.isin(self.reps, reps_test))
                data_test = dict(emg=self.emgs[idxs].copy(),
                                 imu=self.imus[idxs].copy() if self.imu else self.imus,
                                 lbl=self.lbls[idxs].copy())
                print(f"train data: shape=(emg: {data_train['emg'].shape}, imu: {data_train['imu'].shape}), train target: shape={data_train['lbl'].shape}")
                print(f"test data: shape=(emg: {data_test['emg'].shape}, imu: {data_test['imu'].shape}), test target: shape={data_test['lbl'].shape}")
                print(f"val data: shape=(emg: {data_val['emg'].shape}, imu: {data_val['imu'].shape}), val target: shape={data_val['lbl'].shape}")
                print("# Saving processed data...")
                np.savez(str(path_save / f"train_fold{rep}.npz"), **data_train)
                np.savez(str(path_save / f"test_fold{rep}.npz"), **data_test)
                np.savez(str(path_save / f"val_fold{rep}.npz"), **data_val)
                # release memory
                del data_train, data_val, data_test
                gc.collect()
        elif mode == "subject":
            reps_unique = np.unique(self.reps)[1:]  # discard session 0
            subs_unique = np.unique(self.subs)
            for sub in subs_unique:
                print(f"Subject #{sub}...")
                # get repetitions for train/val/test sets...
                subs_trainval = subs_unique[np.where(np.isin(subs_unique, sub, invert=True))]
                subs_val = subs_trainval[0 if sub == len(subs_unique) else (sub - 1)]
                subs_train = np.setdiff1d(subs_trainval, subs_val)
                subs_test = sub
                # split train set
                sub_idxs = np.where(np.isin(self.subs, subs_train))
                idxs = np.intersect1d(sub_idxs, np.where(np.isin(self.reps, reps_unique)))
                data_train = dict(emg=self.emgs[idxs].copy(),
                                  imu=self.imus[idxs].copy() if self.imu else self.imus,
                                  lbl=self.lbls[idxs].copy())
                # split val set
                sub_idxs = np.where(np.isin(self.subs, subs_val))
                idxs = np.intersect1d(sub_idxs, np.where(np.isin(self.reps, reps_unique)))
                data_val = dict(emg=self.emgs[idxs].copy(),
                                imu=self.imus[idxs].copy() if self.imu else self.imus,
                                lbl=self.lbls[idxs].copy())
                # split test set
                sub_idxs = np.where(np.isin(self.subs, subs_test))
                idxs = np.intersect1d(sub_idxs, np.where(np.isin(self.reps, reps_unique)))
                data_test = dict(emg=self.emgs[idxs].copy(),
                                 imu=self.imus[idxs].copy() if self.imu else self.imus,
                                 lbl=self.lbls[idxs].copy())
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
            reps_unique = np.unique(self.reps)[1:]  # discard session 0
            subs_unique = np.unique(self.subs)
            for sub in subs_unique:
                print(f"Subject #{sub}...")
                # get repetitions for train/val/test sets...
                subs_trainval = subs_unique[np.where(np.isin(subs_unique, sub, invert=True))]
                subs_val = subs_trainval[0 if sub == len(subs_unique) else (sub - 1)]
                subs_train = np.setdiff1d(subs_trainval, subs_val)
                subs_test = sub
                # split train set
                sub_idxs = np.where(np.isin(self.subs, subs_train))
                idxs = np.intersect1d(sub_idxs, np.where(np.isin(self.reps, reps_unique)))
                data_train = dict(emg=self.emgs[idxs].copy(),
                                  imu=self.imus[idxs].copy() if self.imu else self.imus,
                                  lbl=self.lbls[idxs].copy())
                # split val set
                sub_idxs = np.where(np.isin(self.subs, subs_val))
                idxs = np.intersect1d(sub_idxs, np.where(np.isin(self.reps, reps_unique)))
                data_val = dict(emg=self.emgs[idxs].copy(),
                                imu=self.imus[idxs].copy() if self.imu else self.imus,
                                lbl=self.lbls[idxs].copy())
                # split test set
                sub_idxs = np.where(np.isin(self.subs, subs_test))
                idxs = np.intersect1d(sub_idxs, np.where(np.isin(self.reps, reps_unique)))
                data_test = dict(emg=self.emgs[idxs].copy(),
                                 imu=self.imus[idxs].copy() if self.imu else self.imus,
                                 lbl=self.lbls[idxs].copy())
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