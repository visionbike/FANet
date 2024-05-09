from abc import abstractmethod, ABC

__all__ = ["PreprocessorBase"]


class PreprocessorBase(ABC):
    """
    The abstract processor implementation.
    """

    def __init__(self):
        pass

    @abstractmethod
    def load_data(self) -> None:
        pass

    @abstractmethod
    def process_data(self) -> None:
        pass

    @abstractmethod
    def split_data(self, mode: str, save_path: str, valtest_reps: list = None) -> None:
        """
        Splits the following the mode.

        :param mode: splitting mode.
        :param save_path: the saving path for the split data.
        :param valtest_reps: the repetition indices used for the validation and test sets. Default: None.
        """
        pass
