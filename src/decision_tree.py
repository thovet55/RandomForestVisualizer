from typing import Any, Optional

import numpy as np

from .tree_node import TreeNode

class DecisionTreeClassifier:
    def __init__(
        self,
        max_depth: Optional[int] = 3,
        min_samples_split: int = 2,
        max_features: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        if max_depth is not None and max_depth < 0:
            raise ValueError("Max_depth must be non-negative or None.")

        if min_samples_split < 2:
            raise ValueError("Min_samples_split must be at least 2.")

        if max_features is not None and max_features < 1:
            raise ValueError("Max_features must be at least 1 or None.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

        self.root: Optional[TreeNode] = None
        self.n_features_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None

        self.events = []
        self._next_node_id = 0
        self._rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y)

        if X_array.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if y_array.ndim != 1:
            raise ValueError("y must be a 1D array.")

        if len(X_array) != len(y_array):
            raise ValueError("X and y must have the same length.")

        if len(X_array) == 0:
            raise ValueError("Training data cannot be empty.")

        self.n_features_ = X_array.shape[1]
        self.classes_ = np.unique(y_array)

        self.events = []
        self._next_node_id = 0
        self._rng = np.random.default_rng(self.random_state)

        #build tree
        self.root = self._build_tree(X_array, y_array, depth=0)

        return self

    def _gini(self, y) -> float:
        #纯净度
        if len(y) == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        return float(1.0 - np.sum(probabilities ** 2))

    def _majority_class(self, y) -> Any:
        labels, counts = np.unique(y, return_counts=True)
        best_index = np.argmax(counts)
        label = labels[best_index]

        if hasattr(label, "item"):
            return label.item()

        return label

    def _count_classes(self, y) -> dict:
        labels, counts = np.unique(y, return_counts=True)

        result = {}

        for label, count in zip(labels, counts):
            if hasattr(label, "item"):
                label = label.item()

            result[label] = int(count)

        return result

    def _get_next_node_id(self) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    def _get_stop_reason(self, y, depth: int) -> Optional[str]:
        if self.max_depth is not None and depth >= self.max_depth:
            return "max_depth"

        if len(y) < self.min_samples_split:
            return "min_samples_split"

        if len(np.unique(y)) == 1:
            return "pure_node"

        return None

    def _choose_feature_indices(self, n_features: int) -> np.ndarray:
        if self.max_features is None:
            return np.arange(n_features)

        feature_count = min(self.max_features, n_features)

        return self._rng.choice(
            n_features,
            size=feature_count,
            replace=False,
        )

    def _find_best_split(self, X, y, feature_indices):
        best_feature_index = None
        best_threshold = None
        best_gini = float("inf")

        num_samples = len(y)

        for feature_index in feature_indices:
            feature_values = X[:, feature_index]
            unique_values = np.unique(feature_values)

            if len(unique_values) <= 1:
                continue

            thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = feature_values > threshold

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_y = y[left_mask]
                right_y = y[right_mask]

                left_gini = self._gini(left_y)
                right_gini = self._gini(right_y)

                weighted_gini = (
                    len(left_y) / num_samples * left_gini
                    + len(right_y) / num_samples * right_gini
                )

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature_index = int(feature_index)
                    best_threshold = float(threshold)

        return best_feature_index, best_threshold, best_gini

    def _build_tree(self, X, y, depth: int) -> TreeNode:
        node = TreeNode(
            node_id=self._get_next_node_id(),
            depth=depth,
            prediction=self._majority_class(y),
            gini=self._gini(y),
            num_samples=len(y),
            class_counts=self._count_classes(y),
        )

        stop_reason = self._get_stop_reason(y, depth)

        if stop_reason is not None:
            node.is_leaf = True

            self.events.append({
                "event_type": "leaf_created",
                "node_id": node.node_id,
                "depth": depth,
                "reason": stop_reason,
                "prediction": node.prediction,
                "gini": node.gini,
                "num_samples": node.num_samples,
                "class_counts": node.class_counts,
                "message": f"Create leaf node because of {stop_reason}.",
            })

            return node

        feature_indices = self._choose_feature_indices(X.shape[1])

        self.events.append({
            "event_type": "node_split_start",
            "node_id": node.node_id,
            "depth": depth,
            "candidate_features": feature_indices.tolist(),
            "gini": node.gini,
            "num_samples": node.num_samples,
            "class_counts": node.class_counts,
            "message": "Start searching for the best split.",
        })

        best_feature_index, best_threshold, best_gini = self._find_best_split(
            X,
            y,
            feature_indices,
        )

        if best_feature_index is None or best_gini >= node.gini:
            node.is_leaf = True

            self.events.append({
                "event_type": "leaf_created",
                "node_id": node.node_id,
                "depth": depth,
                "reason": "no_valid_split",
                "prediction": node.prediction,
                "gini": node.gini,
                "num_samples": node.num_samples,
                "class_counts": node.class_counts,
                "message": "Create leaf node because no valid split was found.",
            })

            return node

        node.feature_index = best_feature_index
        node.threshold = best_threshold

        self.events.append({
            "event_type": "split_chosen",
            "node_id": node.node_id,
            "depth": depth,
            "feature_index": best_feature_index,
            "threshold": best_threshold,
            "gini_before": node.gini,
            "gini_after": best_gini,
            "message": (
                f"Choose feature {best_feature_index} <= {best_threshold:.3f}."
            ),
        })

        left_mask = X[:, best_feature_index] <= best_threshold
        right_mask = X[:, best_feature_index] > best_threshold

        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def _check_is_fitted(self):
        #是否训练
        if self.root is None:
            raise ValueError("DecisionTreeClassifier has not been fitted yet.")

    def predict_one(self, x):
        prediction, _ = self.predict_one_with_path(x)
        return prediction

    def predict_one_with_path(self, x):
        self._check_is_fitted()

        x_array = np.asarray(x, dtype=float)

        if x_array.ndim != 1:
            raise ValueError("x must be a 1D array.")

        if len(x_array) != self.n_features_:
            raise ValueError("x has a different number of features.")

        node = self.root
        path = []

        while node is not None:
            path.append(node)

            if node.is_leaf:
                #叶节点
                return node.prediction, path

            if x_array[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right

        raise RuntimeError("Prediction failed because the tree is broken.")

    def predict(self, X):
        self._check_is_fitted()

        X_array = np.asarray(X, dtype=float)

        if X_array.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X_array.shape[1] != self.n_features_:
            raise ValueError("X has a different number of features.")

        predictions = []

        for x in X_array:
            prediction = self.predict_one(x)
            predictions.append(prediction)

        return np.array(predictions)
