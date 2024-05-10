from typing import Any
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = ["visualize_confusion_matrix"]


def visualize_confusion_matrix(cm: NDArray) -> Any:
    """
    Visualize confusion matrix.

    :param cm: confusion matrix.
    :return: visualization figure.
    """
    cm_df = pd.DataFrame(cm)
    plt.figure(figsize=(20, 14))
    sns.set(font_scale=0.6)
    fig_ = sns.heatmap(cm_df, annot=False, cbar=True).get_figure()
    plt.close(fig_)
    return fig_
