import torchmetrics as tm
import torchmetrics.classification as tmc

__all__ = ["get_metrics"]


def get_metrics(num_classes: int,
                cm: bool = False,
                prefix: str = None,
                postfix: str = None) -> tm.MetricCollection:
    """
    Get the classification metric collection.

    :param num_classes: the number of classes.
    :param cm: the string to append in front of keys of the output dict. Default: None.
    :param prefix: the string to append after keys of the output dict. Default: None.
    :param postfix: the string to append after keys of the output dict. Default: None.
    :return: the classification metric collection.
    """
    metrics = dict(
        accuracy=tmc.MulticlassAccuracy(num_classes, average="micro"),
        balanced_accuracy=tmc.MulticlassRecall(num_classes, average="macro"),
        mathews_corr_coef=tmc.MulticlassMatthewsCorrCoef(num_classes),
        f1_score=tmc.MulticlassF1Score(num_classes, average="macro")
    )
    if cm:
        metrics["confusion_matrix"] = tmc.MulticlassConfusionMatrix(num_classes, normalize="true")
    metrics = tm.MetricCollection(
        metrics=metrics,
        prefix=prefix,
        postfix=postfix,
        compute_groups=True
    )
    return metrics
