import numpy as np

__all__ = [
    "compute_class_weights",
    "compute_samples_per_class",
    "get_class_weights",
    "get_samples_per_class"
]


def compute_class_weights(x: np.ndarray) -> np.ndarray:
    """
    Compute the class weights.

    :param x: the list of labels.
    :return: the class weights.
    """
    # get tuple with label and corresponding counts
    unique = np.unique(x, return_counts=True)
    class_weights = sum(unique[1]) / (len(unique[0]) * unique[1])
    return class_weights


def compute_samples_per_class(x: np.ndarray) -> np.ndarray:
    """
    Compute number of samples per class.

    :param x: the list of labels.
    :return: the number of samples per class.
    """
    samples_per_class = np.bincount(x)
    return samples_per_class


def get_class_weights(num_classes: int, based_weights: str | float | list = None) -> None | list | float:
    """
    Get the clas weights from configuration.

    :param num_classes: the number of classes.
    :param based_weights: the weight file path or weight array.
        If "base_weights" is *.npy file path, load the class weights from the file.
        If "base_weights" is float or list of float value, return the value itself.
        If "base_weights" is None, return None.
    :return: the class weight array or None.
    """
    if isinstance(based_weights, str):
        weights = np.load(based_weights).tolist()
    elif isinstance(based_weights, float):
        weights = [based_weights] * num_classes
    else:
        weights = based_weights
    return weights


def get_samples_per_class(based_samples_per_class: str | list = None) -> None | list:
    """
    Get the clas weights form configuration.

    :param (str, list, optional) based_samples_per_class: the samples_per_class file path or samples_per_class array.
        If based_samples_per_class is *. npy file path, load the samples_per_class from the file.
        If based_samples_per_class is float or list of float value, return the value itself.
        If based_samples_per_class is None, return None. Default: None.
    :return (list, optional): the samples per class array or None.
    """
    if isinstance(based_samples_per_class, str):
        samples_per_class = np.load(based_samples_per_class).tolist()
    else:
        samples_per_class = based_samples_per_class
    return samples_per_class
