from collections import Counter

import numpy as np

from .decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(
        self,
        n_trees=5,
        max_depth=3,
        min_samples_split=2,
        max_features=None,
        random_state=None,
    ):
        if not isinstance(n_trees, int) or n_trees <= 0:
            raise ValueError("n_trees must be a positive integer.")

        if max_depth is not None:
            if not isinstance(max_depth, int) or max_depth < 0:
                raise ValueError("max_depth must be None or a non-negative integer.")

        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer greater than or equal to 2.")

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

        # bootstrap_indices[i] 保存第 i 棵树使用的样本索引。
        self.bootstrap_indices = []
        # 保存每棵树使用的随机种子，方便检查和复现实验。
        # 相同的种子产生结果相同
        self.tree_random_states = []
        self.classes_ = None
        self.n_features_in_ = None
        self.is_fitted_ = False

    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)

        self._validate_training_data(X, y)

        number_of_samples = X.shape[0]
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        # 每次调用 fit 都重新训练整个森林。
        self.trees = []
        self.bootstrap_indices = []
        self.tree_random_states = []

        random_generator = np.random.default_rng(self.random_state)
        used_tree_seeds = set()

        for _ in range(self.n_trees):
            # 从 0 到 number_of_samples - 1 中有放回抽取。
            # 抽取数量与原训练集样本数量相同。
            sample_indices = random_generator.choice(
                number_of_samples,
                size=number_of_samples,
                replace=True,
            )

            bootstrap_X = X[sample_indices]
            bootstrap_y = y[sample_indices]

            # 保证每棵树使用不同的随机种子。
            tree_seed = self._generate_unique_tree_seed(
                random_generator,
                used_tree_seeds,
            )
            used_tree_seeds.add(tree_seed)

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=tree_seed,
            )

            tree.fit(bootstrap_X, bootstrap_y)

            self.trees.append(tree)
            self.bootstrap_indices.append(sample_indices.copy())
            self.tree_random_states.append(tree_seed)

        self.is_fitted_ = True
        return self

    def predict(self, X):

        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must contain {self.n_features_in_} features per sample."
            )

        predictions = [self.predict_one(sample) for sample in X]
        return np.asarray(predictions)

    def predict_one(self, x):

        self._check_is_fitted()
        x = self._validate_single_sample(x)

        tree_predictions = []

        for tree in self.trees:
            prediction = tree.predict_one(x)
            tree_predictions.append(prediction)

        final_prediction, _ = self._majority_vote(tree_predictions)
        return final_prediction

    def predict_one_with_votes(self, x):

        self._check_is_fitted()
        x = self._validate_single_sample(x)

        tree_predictions = []
        tree_paths = []

        for tree in self.trees:
            prediction, path = tree.predict_one_with_path(x)

            tree_predictions.append(prediction)
            tree_paths.append(path)

        final_prediction, vote_counts = self._majority_vote(tree_predictions)

        return {
            "tree_predictions": tree_predictions,
            "tree_paths": tree_paths,
            "vote_counts": vote_counts,
            "final_prediction": final_prediction,
        }

    @staticmethod
    def _majority_vote(predictions):
        if len(predictions) == 0:
            raise ValueError("predictions cannot be empty.")

        vote_counter = Counter(predictions)
        highest_vote_count = max(vote_counter.values())

        tied_classes = [
            label
            for label, count in vote_counter.items()
            if count == highest_vote_count
        ]

        final_prediction = min(tied_classes)

        return final_prediction, dict(vote_counter)

    @staticmethod
    def _generate_unique_tree_seed(random_generator, used_seeds):
        maximum_seed = np.iinfo(np.int32).max

        while True:
            tree_seed = int(random_generator.integers(0, maximum_seed))

            if tree_seed not in used_seeds:
                return tree_seed

    @staticmethod
    def _validate_training_data(X, y):
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        if y.ndim != 1:
            raise ValueError("y must be a one-dimensional array.")

        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")

        if len(X) == 0:
            raise ValueError("Training data cannot be empty.")

        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")

    def _validate_single_sample(self, x):
        x = np.asarray(x)

        if x.ndim != 1:
            raise ValueError("x must be a one-dimensional array.")

        if len(x) != self.n_features_in_:
            raise ValueError(
                f"x must contain exactly {self.n_features_in_} features."
            )

        return x

    def _check_is_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError(
                "The random forest has not been fitted. Call fit(X, y) first."
            )
