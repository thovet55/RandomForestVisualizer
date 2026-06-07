from src.data_utils import (
    make_toy_classification_data,
    train_test_split,
)
from src.metrics import accuracy_score
from src.random_forest import RandomForestClassifier


def main():
    """
    运行 Random Forest 的命令行示例。

    main.py 不负责复杂界面，只用于检查：
    1. 数据能否生成和划分。
    2. Random Forest 能否完成训练。
    3. 模型能否进行预测。
    4. 每棵树的路径和投票信息能否正常返回。
    """

    # 生成两个数值特征组成的分类数据集。
    X, y = make_toy_classification_data(seed=42)

    # 使用固定随机种子，使每次运行得到相同的数据划分。
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_ratio=0.3,
        seed=42,
    )

    # max_features=1 表示每个节点分裂时，
    # 只随机选择一个特征作为候选特征。
    forest = RandomForestClassifier(
        n_trees=5,
        max_depth=3,
        min_samples_split=2,
        max_features=1,
        random_state=42,
    )

    print("Training Random Forest...")
    forest.fit(X_train, y_train)

    # 对整个测试集进行预测。
    y_pred = forest.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n=== Test Results ===")
    print("True labels:")
    print(y_test)

    print("\nPredicted labels:")
    print(y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")

    # 选择第一个测试样本，展示森林内部的投票过程。
    selected_sample_index = 0
    selected_sample = X_test[selected_sample_index]
    selected_true_label = y_test[selected_sample_index]

    vote_details = forest.predict_one_with_votes(
        selected_sample
    )

    print("\n=== Single Sample Voting Details ===")
    print(f"Test sample index: {selected_sample_index}")
    print(f"Feature values: {selected_sample}")
    print(f"True label: {selected_true_label}")

    tree_predictions = vote_details["tree_predictions"]
    tree_paths = vote_details["tree_paths"]

    # 分别输出每棵树的预测结果和路径长度。
    for tree_index, (prediction, path) in enumerate(
        zip(tree_predictions, tree_paths)
    ):
        print(
            f"Tree {tree_index + 1}: "
            f"prediction={prediction}, "
            f"path length={len(path)}"
        )

    print("\nVote counts:")
    print(vote_details["vote_counts"])

    print("\nFinal prediction:")
    print(vote_details["final_prediction"])

    # Bootstrap 索引可以用于之后的采样可视化。
    print("\n=== Bootstrap Sample Indices ===")

    for tree_index, sample_indices in enumerate(
        forest.bootstrap_indices
    ):
        print(
            f"Tree {tree_index + 1}: {sample_indices}"
        )


if __name__ == "__main__":
    main()
