import lightning as l

__all__ = ['get_dataloader']


def get_dataloader(mode: str, data_name: str, **kwargs) -> l.LightningDataModule:
    """

    :param mode: dataloader mode for different evaluation:
        * "trainvaltest": data is split to train/val/test sets.
        * "session": data is split to inter-session cross validation.
        * "subject": data is split to inter-subject cross validation.
        * "subject-adaptive": data is split to subject-adaptive transfer cross validation.
    :param data_name: the dataset name.
    :param kwargs: dataloader arguments.
    :return: the dataloader.
    """
    if data_name == "asr":
        from .dataset import DatasetAsr
        dataset_ = DatasetAsr
    elif data_name == "nina5":
        from .dataset import DatasetNina5
        dataset_ = DatasetNina5
    else:
        raise ValueError(f"Invalid dataset name, got {data_name}.")

    if mode == "trainvaltest":
        from .dataloader import DataloaderBase
        return DataloaderBase(dataset_, **kwargs)
    elif mode == "session":
        from .dataloader import DataloaderSession
        return DataloaderSession(dataset_, **kwargs)
    elif mode == "subject":
        from .dataloader import DataloaderSubject
        return DataloaderSubject(dataset_, **kwargs)
    elif mode == "subject-adaptive-transfer":
        from .dataloader import DataloaderSubjectAdaptiveTransfer
        return DataloaderSubjectAdaptiveTransfer(dataset_, **kwargs)
    else:
        raise ValueError(f"Invalid dataloader mode, got {mode}.")
