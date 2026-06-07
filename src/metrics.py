from typing import Any, Sequence

import numpy as np


def _validate_targets(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """检查并转换真实标签和预测标签。"""
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    if y_true_array.ndim != 1 or y_pred_array.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional.")

    if len(y_true_array) != len(y_pred_array):
        raise ValueError("y_true and y_pred must have the same length.")

    if len(y_true_array) == 0:
        raise ValueError("y_true and y_pred cannot be empty.")

    return y_true_array, y_pred_array


def accuracy_score(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> float:
    """计算分类准确率。"""
    y_true_array, y_pred_array = _validate_targets(y_true, y_pred)

    correct_count = np.sum(y_true_array == y_pred_array)
    return float(correct_count / len(y_true_array))


def confusion_matrix(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> np.ndarray:
    """
    计算混淆矩阵。

    矩阵的行表示真实类别，列表示预测类别。
    类别按照从小到大的顺序排列。
    """
    y_true_array, y_pred_array = _validate_targets(y_true, y_pred)

    labels = np.unique(np.concatenate((y_true_array, y_pred_array)))
    label_to_index = {
        label: index for index, label in enumerate(labels)
    }

    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for true_label, predicted_label in zip(y_true_array, y_pred_array):
        true_index = label_to_index[true_label]
        predicted_index = label_to_index[predicted_label]
        matrix[true_index, predicted_index] += 1

    return matrix
