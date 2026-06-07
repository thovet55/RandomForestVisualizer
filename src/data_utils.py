from typing import Tuple

import numpy as np


def train_test_split(
    X,
    y,
    test_ratio: float = 0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将数据集随机划分为训练集和测试集。

    参数：
    - X: 特征数据，可以是 list 或 NumPy array
    - y: 标签数据，可以是 list 或 NumPy array
    - test_ratio: 测试集比例，例如 0.3 表示 30% 数据作为测试集
    - seed: 随机种子，用于保证每次划分结果可复现

    返回：
    - X_train, X_test, y_train, y_test
    """
    X_array = np.asarray(X)
    y_array = np.asarray(y)

    if len(X_array) != len(y_array):
        raise ValueError("X and y must have the same length.")

    if len(X_array) < 2:
        raise ValueError("Dataset must contain at least 2 samples.")

    if test_ratio <= 0 or test_ratio >= 1:
        raise ValueError("test_ratio must be between 0 and 1.")

    num_samples = len(X_array)

    rng = np.random.default_rng(seed)

    indices = np.arange(num_samples)
    rng.shuffle(indices)

    test_size = int(num_samples * test_ratio)

    if test_size == 0:
        test_size = 1

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train = X_array[train_indices]
    X_test = X_array[test_indices]
    y_train = y_array[train_indices]
    y_test = y_array[test_indices]

    return X_train, X_test, y_train, y_test


def make_toy_classification_data(
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成一个二维 toy classification 数据集。

    数据集包含两个类别：
    - 类别 0：点大多分布在左下方
    - 类别 1：点大多分布在右上方

    返回：
    - X: 二维特征数据，形状为 (样本数, 2)
    - y: 分类标签，形状为 (样本数,)
    """
    rng = np.random.default_rng(seed)

    num_samples_per_class = 30

    class_0 = rng.normal(
        loc=[1.5, 1.5],
        scale=0.5,
        size=(num_samples_per_class, 2),
    )

    class_1 = rng.normal(
        loc=[4.0, 4.0],
        scale=0.5,
        size=(num_samples_per_class, 2),
    )

    X = np.vstack((class_0, class_1))

    y_class_0 = np.zeros(num_samples_per_class, dtype=int)
    y_class_1 = np.ones(num_samples_per_class, dtype=int)
    y = np.concatenate((y_class_0, y_class_1))

    indices = np.arange(len(X))
    rng.shuffle(indices)

    X = X[indices]
    y = y[indices]

    return X, y
