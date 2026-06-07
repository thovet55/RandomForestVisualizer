import matplotlib.pyplot as plt
import numpy as np


def _validate_2d_data(X, y):
    """
    检查数据是否适合二维可视化。

    Random Forest 算法本身可以处理更多特征，
    但当前可视化版本只绘制两个数值特征。
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")

    if X.shape[1] != 2:
        raise ValueError("Visualization requires exactly two features.")

    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional array.")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")

    if len(X) == 0:
        raise ValueError("X and y cannot be empty.")

    return X, y


def _create_figure(ax=None, figsize=(8, 6)):
    """
    创建 Matplotlib figure 和 axes。

    如果 app.py 已经传入 axes，就直接使用已有 axes。
    """
    if ax is None:
        figure, ax = plt.subplots(figsize=figsize)
    else:
        figure = ax.figure

    return figure, ax


def _get_event_value(event, key, default=None):
    """
    从 VisualizationEvent 或字典事件中读取数据。

    事件的固定字段保存在对象本身，其他数据保存在 data 中。
    同时支持对象和字典，可以让 app.py 调用更灵活。
    """
    if event is None:
        return default

    if isinstance(event, dict):
        if key in event:
            return event[key]

        event_data = event.get("data", {})
        return event_data.get(key, default)

    if hasattr(event, key):
        value = getattr(event, key)

        if value is not None:
            return value

    event_data = getattr(event, "data", {})
    return event_data.get(key, default)


def _draw_class_points(ax, X, y, size=60, alpha=0.8):
    """
    按类别绘制数据点。

    每个类别使用不同颜色，并自动创建图例。
    """
    class_labels = np.unique(y)
    color_map = plt.get_cmap("tab10")

    for class_position, class_label in enumerate(class_labels):
        class_mask = y == class_label

        ax.scatter(
            X[class_mask, 0],
            X[class_mask, 1],
            s=size,
            alpha=alpha,
            color=color_map(class_position % 10),
            edgecolors="black",
            linewidths=0.5,
            label=f"Class {class_label}",
        )


def _add_message(ax, message):
    """
    在图的左下角显示当前算法步骤说明。
    """
    if not message:
        return

    ax.text(
        0.02,
        0.02,
        message,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )


def plot_dataset(
    X,
    y,
    title="Two-Dimensional Dataset",
    message="",
    ax=None,
):
    """
    绘制二维分类数据集。

    Parameters
    ----------
    X : array-like
        两个数值特征组成的数据。

    y : array-like
        每个样本的类别标签。

    title : str
        图表标题。

    message : str
        当前步骤的解释文字。

    ax : matplotlib.axes.Axes or None
        可选的 Matplotlib axes。

    Returns
    -------
    tuple
        Matplotlib figure 和 axes。
    """
    X, y = _validate_2d_data(X, y)
    figure, ax = _create_figure(ax)

    _draw_class_points(ax, X, y)

    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)

    _add_message(ax, message)

    figure.tight_layout()
    return figure, ax


def plot_bootstrap_sample(
    X,
    y,
    event,
    title="Bootstrap Sampling",
    ax=None,
):
    """
    绘制 Bootstrap 有放回采样结果。

    event.data 应包含：
        sample_indices：当前树抽中的原始样本索引。

    被抽中的样本会高亮显示。
    同一个样本被重复抽中时，点会变大并显示重复次数。
    """
    X, y = _validate_2d_data(X, y)
    figure, ax = _create_figure(ax)

    sample_indices = _get_event_value(event, "sample_indices")

    if sample_indices is None:
        raise ValueError(
            "Bootstrap event must contain 'sample_indices'."
        )

    sample_indices = np.asarray(sample_indices, dtype=int).reshape(-1)

    if np.any(sample_indices < 0) or np.any(sample_indices >= len(X)):
        raise ValueError("Bootstrap sample indices are out of range.")

    # bincount 可以统计每个原始样本被抽中了多少次。
    sample_counts = np.bincount(
        sample_indices,
        minlength=len(X),
    )

    selected_indices = np.flatnonzero(sample_counts > 0)

    # 所有原始样本先用灰色显示。
    ax.scatter(
        X[:, 0],
        X[:, 1],
        s=45,
        color="lightgray",
        edgecolors="gray",
        alpha=0.7,
        label="Not selected",
    )

    class_labels = np.unique(y)
    color_map = plt.get_cmap("tab10")

    # 将抽中的样本按照类别重新高亮。
    for class_position, class_label in enumerate(class_labels):
        class_indices = selected_indices[
            y[selected_indices] == class_label
        ]

        if len(class_indices) == 0:
            continue

        # 重复抽中的次数越多，点的尺寸越大。
        point_sizes = 80 + 35 * (sample_counts[class_indices] - 1)

        ax.scatter(
            X[class_indices, 0],
            X[class_indices, 1],
            s=point_sizes,
            color=color_map(class_position % 10),
            edgecolors="black",
            linewidths=0.8,
            alpha=0.9,
            label=f"Selected class {class_label}",
        )

    # 被重复抽中的点显示次数，例如 ×2、×3。
    duplicate_indices = np.flatnonzero(sample_counts > 1)

    for sample_index in duplicate_indices:
        ax.annotate(
            f"×{sample_counts[sample_index]}",
            (
                X[sample_index, 0],
                X[sample_index, 1],
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

    tree_index = _get_event_value(event, "tree_index")
    message = _get_event_value(event, "message", "")

    if tree_index is not None:
        title = f"{title} - Tree {tree_index + 1}"

    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)

    _add_message(ax, message)

    figure.tight_layout()
    return figure, ax


def plot_node_split(
    X,
    y,
    event,
    title="Decision Tree Node Split",
    ax=None,
):
    """
    绘制当前节点选中的分割线。

    event.data 应包含：
        feature_index：参与分裂的特征编号。
        threshold：最终选择的阈值。

    event.data 可选包含：
        sample_indices：当前节点中的原始样本索引。
        gini：分裂后的 Gini impurity。
    """
    X, y = _validate_2d_data(X, y)
    figure, ax = _create_figure(ax)

    feature_index = _get_event_value(event, "feature_index")
    threshold = _get_event_value(event, "threshold")
    node_sample_indices = _get_event_value(event, "sample_indices")
    gini = _get_event_value(event, "gini")

    if feature_index not in (0, 1):
        raise ValueError("feature_index must be 0 or 1.")

    if threshold is None:
        raise ValueError("Split event must contain 'threshold'.")

    if node_sample_indices is None:
        _draw_class_points(ax, X, y)
    else:
        node_sample_indices = np.asarray(
            node_sample_indices,
            dtype=int,
        ).reshape(-1)

        if np.any(node_sample_indices < 0) or np.any(
            node_sample_indices >= len(X)
        ):
            raise ValueError("Node sample indices are out of range.")

        # 当前节点之外的样本使用灰色显示。
        ax.scatter(
            X[:, 0],
            X[:, 1],
            s=40,
            color="lightgray",
            alpha=0.45,
            label="Outside current node",
        )

        _draw_class_points(
            ax,
            X[node_sample_indices],
            y[node_sample_indices],
            size=75,
            alpha=0.9,
        )

    # Feature 0 对应横坐标，因此画竖线。
    if feature_index == 0:
        ax.axvline(
            x=threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Feature 0 <= {threshold:.3f}",
        )
    else:
        # Feature 1 对应纵坐标，因此画横线。
        ax.axhline(
            y=threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Feature 1 <= {threshold:.3f}",
        )

    node_id = _get_event_value(event, "node_id")
    message = _get_event_value(event, "message", "")

    if node_id is not None:
        title = f"{title} - Node {node_id}"

    if gini is not None:
        message = f"{message}\nGini after split: {gini:.4f}".strip()

    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)

    _add_message(ax, message)

    figure.tight_layout()
    return figure, ax


def plot_test_point(
    X,
    y,
    test_point,
    event=None,
    title="Test Sample",
    ax=None,
):
    """
    在二维数据集上绘制需要预测的测试点。

    测试点使用黑色星形标记，与训练样本区分。
    """
    X, y = _validate_2d_data(X, y)
    test_point = np.asarray(test_point)

    if test_point.ndim != 1 or len(test_point) != 2:
        raise ValueError(
            "test_point must contain exactly two feature values."
        )

    figure, ax = _create_figure(ax)

    _draw_class_points(ax, X, y)

    ax.scatter(
        test_point[0],
        test_point[1],
        s=220,
        marker="*",
        color="black",
        edgecolors="white",
        linewidths=1,
        label="Test point",
        zorder=5,
    )

    message = _get_event_value(event, "message", "")
    prediction = _get_event_value(event, "prediction")

    if prediction is not None:
        message = f"{message}\nPrediction: {prediction}".strip()

    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)

    _add_message(ax, message)

    figure.tight_layout()
    return figure, ax


def _get_path_step_value(step, key, default=None):
    """
    从路径中的节点对象或字典读取字段。
    """
    if isinstance(step, dict):
        return step.get(key, default)

    return getattr(step, key, default)


def _format_path_step(step, step_index):
    """
    将一个路径节点转换成适合显示的文字。
    """
    is_leaf = _get_path_step_value(step, "is_leaf", False)
    prediction = _get_path_step_value(step, "prediction")
    node_id = _get_path_step_value(step, "node_id", "?")

    if is_leaf:
        return (
            f"Step {step_index + 1}: Node {node_id} is a leaf "
            f"→ predict class {prediction}"
        )

    feature_index = _get_path_step_value(step, "feature_index")
    threshold = _get_path_step_value(step, "threshold")
    direction = _get_path_step_value(step, "direction")

    if feature_index is None or threshold is None:
        return f"Step {step_index + 1}: Visit node {node_id}"

    text = (
        f"Step {step_index + 1}: Node {node_id}, "
        f"feature {feature_index} <= {threshold:.3f}"
    )

    if direction is not None:
        text += f" → go {direction}"

    return text


def plot_prediction_path(
    path,
    prediction=None,
    event=None,
    current_step=None,
    title="Prediction Path",
    ax=None,
):
    """
    用文字面板展示测试样本经过的节点路径。

    Parameters
    ----------
    path : list
        predict_one_with_path() 返回的节点路径。

    prediction : object or None
        当前树的最终预测类别。

    event : VisualizationEvent, dict or None
        当前预测路径事件。

    current_step : int or None
        当前需要高亮的路径步骤编号。
    """
    if path is None or len(path) == 0:
        raise ValueError("Prediction path cannot be empty.")

    if ax is None:
        figure_height = max(4, 1.5 + len(path) * 0.65)
        figure, ax = _create_figure(
            figsize=(9, figure_height)
        )
    else:
        figure, ax = _create_figure(ax)

    if current_step is None:
        current_step = _get_event_value(
            event,
            "path_step_index",
        )

    if prediction is None:
        prediction = _get_event_value(
            event,
            "prediction",
        )

    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=15)

    top_position = 0.92
    line_spacing = 0.72 / max(len(path), 1)

    for step_index, step in enumerate(path):
        step_text = _format_path_step(step, step_index)
        is_current = step_index == current_step

        text_style = {
            "fontsize": 11,
            "verticalalignment": "top",
        }

        if is_current:
            text_style["fontweight"] = "bold"
            text_style["bbox"] = {
                "boxstyle": "round",
                "facecolor": "#fff3b0",
                "edgecolor": "#f0a500",
            }

        ax.text(
            0.05,
            top_position - step_index * line_spacing,
            step_text,
            transform=ax.transAxes,
            **text_style,
        )

    if prediction is not None:
        ax.text(
            0.05,
            0.08,
            f"Tree prediction: {prediction}",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color="darkgreen",
        )

    message = _get_event_value(event, "message", "")

    if message:
        ax.text(
            0.05,
            0.02,
            message,
            transform=ax.transAxes,
            fontsize=10,
            color="dimgray",
        )

    figure.tight_layout()
    return figure, ax


def plot_forest_votes(
    event,
    title="Random Forest Voting",
    ax=None,
):
    """
    绘制森林当前的投票柱状图。

    event.data 应包含：
        vote_counts：类别到票数的字典。

    event.data 可选包含：
        final_prediction：森林的最终预测类别。
    """
    vote_counts = _get_event_value(event, "vote_counts")
    final_prediction = _get_event_value(
        event,
        "final_prediction",
    )

    if not isinstance(vote_counts, dict) or len(vote_counts) == 0:
        raise ValueError(
            "Vote event must contain a non-empty 'vote_counts' dictionary."
        )

    figure, ax = _create_figure(ax)

    class_labels = list(vote_counts.keys())

    # 数值或字符串标签通常可以排序。
    # 如果标签类型不同，则保持字典原来的顺序。
    try:
        class_labels = sorted(class_labels)
    except TypeError:
        pass

    vote_values = [
        vote_counts[class_label]
        for class_label in class_labels
    ]

    bar_colors = []

    for class_label in class_labels:
        if (
            final_prediction is not None
            and class_label == final_prediction
        ):
            bar_colors.append("seagreen")
        else:
            bar_colors.append("steelblue")

    bars = ax.bar(
        [str(label) for label in class_labels],
        vote_values,
        color=bar_colors,
        edgecolor="black",
    )

    # 在每个柱子上方显示具体票数。
    for bar, vote_value in zip(bars, vote_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            str(vote_value),
            horizontalalignment="center",
            verticalalignment="bottom",
        )

    message = _get_event_value(event, "message", "")

    if final_prediction is not None:
        title = f"{title} - Prediction: {final_prediction}"

    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Votes")
    ax.set_title(title)
    ax.set_ylim(
        0,
        max(vote_values) + max(1, max(vote_values) * 0.25),
    )
    ax.grid(axis="y", alpha=0.2)

    _add_message(ax, message)

    figure.tight_layout()
    return figure, ax
def build_tree_prediction_graphviz(
    tree,
    path,
    current_step=None,
):
    """
    Build a Graphviz tree diagram and highlight a prediction path.

    This function does not implement training, Gini calculation, or split search.
    It only reads the existing trained tree structure and prediction path.
    """
    import graphviz

    colors = {
        "current": "#FFD966",
        "visited": "#B6D7A8",
        "future": "#CFE2F3",
        "final_leaf": "#6AA84F",
        "normal": "#E7E6E6",
        "visited_edge": "#6AA84F",
        "current_edge": "#F6B26B",
        "normal_edge": "#999999",
    }

    def get_value(obj, key, default=None):
        if obj is None:
            return default

        if isinstance(obj, dict):
            if key in obj:
                return obj[key]

            data = obj.get("data")
            if isinstance(data, dict) and key in data:
                return data[key]

            return default

        return getattr(obj, key, default)

    def get_root_node(model_or_node):
        for root_name in ("root", "root_node", "root_", "tree", "tree_"):
            root_node = get_value(model_or_node, root_name, None)

            if root_node is not None:
                return root_node

        looks_like_node = (
            get_value(model_or_node, "left", None) is not None
            or get_value(model_or_node, "right", None) is not None
            or get_value(model_or_node, "is_leaf", None) is not None
            or get_value(model_or_node, "prediction", None) is not None
            or get_value(model_or_node, "feature_index", None) is not None
        )

        if looks_like_node:
            return model_or_node

        return None

    def extract_path_node(path_item):
        if isinstance(path_item, dict):
            for node_key in (
                "node",
                "tree_node",
                "current_node",
                "node_obj",
                "node_object",
            ):
                node = path_item.get(node_key)

                if node is not None:
                    return node

            data = path_item.get("data")
            if isinstance(data, dict):
                for node_key in (
                    "node",
                    "tree_node",
                    "current_node",
                    "node_obj",
                    "node_object",
                ):
                    node = data.get(node_key)

                    if node is not None:
                        return node

            return path_item

        return path_item

    def get_node_key(node):
        node_id = get_value(node, "node_id", None)

        if node_id is not None:
            return f"node_{node_id}"

        return f"object_{id(node)}"

    def format_float(value):
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return str(value)

    def build_node_label(node):
        left_child = get_value(node, "left", None)
        right_child = get_value(node, "right", None)

        is_leaf = get_value(node, "is_leaf", None)
        if is_leaf is None:
            is_leaf = left_child is None and right_child is None

        node_id = get_value(node, "node_id", None)

        if is_leaf:
            if node_id is None:
                label_lines = ["Leaf"]
            else:
                label_lines = [f"Leaf {node_id}"]

            prediction = get_value(node, "prediction", None)
            if prediction is not None:
                label_lines.append(f"Predict: {prediction}")
        else:
            if node_id is None:
                label_lines = ["Node"]
            else:
                label_lines = [f"Node {node_id}"]

            feature_index = get_value(node, "feature_index", None)
            threshold = get_value(node, "threshold", None)

            if feature_index is not None and threshold is not None:
                label_lines.append(
                    f"feature_{feature_index} <= {format_float(threshold)}"
                )

        gini = get_value(node, "gini", None)
        if gini is not None:
            label_lines.append(f"Gini: {format_float(gini)}")

        num_samples = get_value(node, "num_samples", None)
        if num_samples is not None:
            label_lines.append(f"Samples: {num_samples}")

        return "\n".join(label_lines)

    if path is None:
        raw_path = []
    elif isinstance(path, dict):
        raw_path = [path]
    else:
        raw_path = list(path)

    path_keys = []
    for path_item in raw_path:
        path_node = extract_path_node(path_item)

        if path_node is not None:
            path_keys.append(get_node_key(path_node))

    path_positions = {}
    for path_index, path_key in enumerate(path_keys):
        if path_key not in path_positions:
            path_positions[path_key] = path_index

    if path_keys and current_step is not None:
        try:
            clamped_step = int(current_step)
        except (TypeError, ValueError):
            clamped_step = 0

        clamped_step = max(0, min(clamped_step, len(path_keys) - 1))
    else:
        clamped_step = None

    path_edge_positions = {}
    for path_index in range(len(path_keys) - 1):
        edge_key = (path_keys[path_index], path_keys[path_index + 1])
        path_edge_positions[edge_key] = path_index

    def get_node_fill_color(node_key):
        if node_key not in path_positions:
            return colors["normal"]

        path_index = path_positions[node_key]
        final_path_index = len(path_keys) - 1

        if clamped_step is None:
            if path_index == final_path_index:
                return colors["final_leaf"]

            return colors["visited"]

        if path_index == final_path_index and path_index <= clamped_step:
            return colors["final_leaf"]

        if path_index == clamped_step:
            return colors["current"]

        if path_index < clamped_step:
            return colors["visited"]

        return colors["future"]

    def get_edge_color(parent_key, child_key):
        edge_index = path_edge_positions.get((parent_key, child_key))

        if edge_index is None:
            return colors["normal_edge"]

        if clamped_step is None:
            return colors["visited_edge"]

        if edge_index < clamped_step:
            return colors["visited_edge"]

        if edge_index == clamped_step:
            return colors["current_edge"]

        return colors["normal_edge"]

    root_node = get_root_node(tree)

    if root_node is None:
        raise ValueError(
            "The selected tree does not contain a root node. "
            "Expected one of: root, root_node, root_, tree, tree_."
        )

    graph = graphviz.Digraph()
    graph.attr(rankdir="TB")
    graph.attr("node", shape="box", style="rounded,filled", fontname="Arial")
    graph.attr("edge", fontname="Arial")

    added_nodes = set()

    def add_node_and_children(node):
        if node is None:
            return

        node_key = get_node_key(node)

        if node_key in added_nodes:
            return

        added_nodes.add(node_key)

        graph.node(
            node_key,
            label=build_node_label(node),
            fillcolor=get_node_fill_color(node_key),
            color="#666666",
        )

        left_child = get_value(node, "left", None)
        right_child = get_value(node, "right", None)

        if left_child is not None:
            left_key = get_node_key(left_child)
            add_node_and_children(left_child)

            edge_color = get_edge_color(node_key, left_key)
            graph.edge(
                node_key,
                left_key,
                label="True / Left",
                color=edge_color,
                penwidth="3" if edge_color != colors["normal_edge"] else "1",
            )

        if right_child is not None:
            right_key = get_node_key(right_child)
            add_node_and_children(right_child)

            edge_color = get_edge_color(node_key, right_key)
            graph.edge(
                node_key,
                right_key,
                label="False / Right",
                color=edge_color,
                penwidth="3" if edge_color != colors["normal_edge"] else "1",
            )

    add_node_and_children(root_node)

    return graph
