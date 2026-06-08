"""
Streamlit app: 自实现 Random Forest 全流程可视化器。

功能：
1. 自己生成 two-moons 风格二维分类数据；
2. 从 src 调用自己实现的 Decision Tree 和 Random Forest；
3. 用动画展示建树、森林边界和多树投票；
4. 在 Streamlit 中用上一帧 / 下一帧按钮逐帧展示完整动画。

运行：
    .venv\\Scripts\\Activate.ps1
    streamlit run app.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.animation import FuncAnimation, PillowWriter

from src.decision_tree import DecisionTreeClassifier
from src.random_forest import RandomForestClassifier
from src.tree_node import TreeNode


# ============================================================
# 1. 自己生成 two-moons 风格二维分类数据，不调用 sklearn
# ============================================================


def make_two_moons(
    n_samples: int = 260,
    noise: float = 0.25,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a simple two-moons binary classification dataset.

    这是 sklearn.datasets.make_moons 的教学版替代实现。
    返回：
        X: shape = [n_samples, 2]
        y: shape = [n_samples]
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    if noise < 0:
        raise ValueError("noise must be non-negative.")

    rng = np.random.default_rng(random_state)

    n_first = n_samples // 2
    n_second = n_samples - n_first

    theta_first = rng.uniform(0.0, np.pi, size=n_first)
    theta_second = rng.uniform(0.0, np.pi, size=n_second)

    first_moon = np.column_stack(
        [
            np.cos(theta_first),
            np.sin(theta_first),
        ]
    )

    second_moon = np.column_stack(
        [
            1.0 - np.cos(theta_second),
            0.5 - np.sin(theta_second),
        ]
    )

    X = np.vstack([first_moon, second_moon])
    y = np.array([0] * n_first + [1] * n_second, dtype=int)

    if noise > 0:
        X = X + rng.normal(loc=0.0, scale=noise, size=X.shape)

    order = rng.permutation(n_samples)
    return X[order], y[order]


# ============================================================
# 2. 模型实现从 src 导入，app.py 只保留界面和可视化逻辑
# ============================================================


# ============================================================
# 3. 画图工具：分类边界、树结构、投票柱状图
# ============================================================


@dataclass
class DemoState:
    X: np.ndarray
    y: np.ndarray
    forest: RandomForestClassifier
    tree1_build_models: List[DecisionTreeClassifier]
    new_sample: np.ndarray
    tree_votes_for_new_sample: np.ndarray
    final_vote_counts: np.ndarray
    final_prediction: int
    prob_class_0: float
    prob_class_1: float
    xx: np.ndarray
    yy: np.ndarray
    grid: np.ndarray
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    max_depth: int
    n_trees: int
    total_frames: int
    phase_1_frames: int
    phase_2_frames: int
    phase_3_frames: int


FEATURE_NAMES = ["Feature 1", "Feature 2"]
CLASS_NAMES = ["Class 0", "Class 1"]


def normalize_max_features(
    max_features: Union[int, str, None],
    n_features: int,
) -> Optional[int]:
    """Convert sidebar max_features options into the integer format used by src."""
    if max_features is None:
        return None
    if max_features == "sqrt":
        return max(1, int(np.sqrt(n_features)))
    if max_features == "log2":
        return max(1, int(np.log2(n_features)))
    return int(max_features)


def build_demo_state(
    n_samples: int,
    noise: float,
    data_seed: int,
    n_trees: int,
    max_depth: int,
    min_samples_split: int,
    max_features: Union[int, str, None],
    forest_seed: int,
    new_sample_x: float,
    new_sample_y: float,
    grid_resolution: int,
) -> DemoState:
    X, y = make_two_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=data_seed,
    )
    model_max_features = normalize_max_features(max_features, n_features=X.shape[1])

    forest = RandomForestClassifier(
        n_trees=n_trees,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=model_max_features,
        random_state=forest_seed,
    )
    forest.fit(X, y)

    tree1_bootstrap_indices = forest.bootstrap_indices[0]
    X_tree1_bootstrap = X[tree1_bootstrap_indices]
    y_tree1_bootstrap = y[tree1_bootstrap_indices]

    tree1_random_state = (
        forest.tree_random_states[0]
        if getattr(forest, "tree_random_states", None)
        else forest_seed
    )

    tree1_build_models: List[DecisionTreeClassifier] = []
    for depth in range(1, max_depth + 1):
        model = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_split=min_samples_split,
            max_features=model_max_features,
            random_state=tree1_random_state,
        )
        model.fit(X_tree1_bootstrap, y_tree1_bootstrap)
        tree1_build_models.append(model)

    x_min, x_max = float(X[:, 0].min() - 0.6), float(X[:, 0].max() + 0.6)
    y_min, y_max = float(X[:, 1].min() - 0.6), float(X[:, 1].max() + 0.6)

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    new_sample = np.array([[new_sample_x, new_sample_y]], dtype=float)
    tree_votes_for_new_sample = np.array(
        [tree.predict(new_sample)[0] for tree in forest.trees],
        dtype=int,
    )

    final_vote_counts = np.bincount(tree_votes_for_new_sample, minlength=2)
    final_prediction = int(forest.predict_one(new_sample[0]))
    prob_class_0 = float(final_vote_counts[0] / n_trees)
    prob_class_1 = float(final_vote_counts[1] / n_trees)

    phase_1_frames = max_depth
    phase_2_frames = n_trees
    phase_3_frames = 1 + n_trees + 1
    total_frames = phase_1_frames + phase_2_frames + phase_3_frames

    return DemoState(
        X=X,
        y=y,
        forest=forest,
        tree1_build_models=tree1_build_models,
        new_sample=new_sample,
        tree_votes_for_new_sample=tree_votes_for_new_sample,
        final_vote_counts=final_vote_counts,
        final_prediction=final_prediction,
        prob_class_0=prob_class_0,
        prob_class_1=prob_class_1,
        xx=xx,
        yy=yy,
        grid=grid,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        max_depth=max_depth,
        n_trees=n_trees,
        total_frames=total_frames,
        phase_1_frames=phase_1_frames,
        phase_2_frames=phase_2_frames,
        phase_3_frames=phase_3_frames,
    )


def draw_boundary(
    ax: plt.Axes,
    demo: DemoState,
    model_or_predictor: Union[DecisionTreeClassifier, RandomForestClassifier, Callable[[np.ndarray], np.ndarray]],
    title: str,
) -> None:
    """Draw a model's decision boundary on the two-dimensional grid."""
    if callable(model_or_predictor) and not hasattr(model_or_predictor, "predict"):
        Z = model_or_predictor(demo.grid)
    else:
        Z = model_or_predictor.predict(demo.grid)  # type: ignore[union-attr]

    Z = np.asarray(Z).reshape(demo.xx.shape)

    ax.contourf(demo.xx, demo.yy, Z, alpha=0.35)
    ax.scatter(
        demo.X[:, 0],
        demo.X[:, 1],
        c=demo.y,
        edgecolor="k",
        s=35,
    )
    ax.set_xlim(demo.x_min, demo.x_max)
    ax.set_ylim(demo.y_min, demo.y_max)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(title)


def draw_tree_structure(
    ax: plt.Axes,
    tree: DecisionTreeClassifier,
    title: str,
) -> None:
    """
    Draw a small binary tree using Matplotlib.

    This replaces sklearn.tree.plot_tree. It is intentionally simple because
    max_depth is small in this teaching visualization.
    """
    if tree.root is None:
        ax.text(0.5, 0.5, "Tree is not fitted", ha="center", va="center")
        ax.axis("off")
        return

    positions: Dict[int, Tuple[float, float]] = {}
    leaf_counter = {"value": 0}

    def assign_positions(node: TreeNode) -> float:
        if node.is_leaf or (node.left is None and node.right is None):
            x = float(leaf_counter["value"])
            leaf_counter["value"] += 1
        else:
            child_x_values = []
            if node.left is not None:
                child_x_values.append(assign_positions(node.left))
            if node.right is not None:
                child_x_values.append(assign_positions(node.right))
            x = float(np.mean(child_x_values))

        y = -float(node.depth)
        positions[node.node_id] = (x, y)
        return x

    assign_positions(tree.root)
    leaf_count = max(1, leaf_counter["value"])
    max_x = float(max(x for x, _ in positions.values()))

    def normalized_position(node: TreeNode) -> Tuple[float, float]:
        x, y = positions[node.node_id]
        if leaf_count == 1:
            x_norm = 0.5
        else:
            x_norm = 0.1 + 0.8 * (x / max_x if max_x > 0 else 0.5)
        y_norm = 0.9 - 0.8 * (node.depth / max(tree.max_depth, 1))
        return x_norm, y_norm

    def node_text(node: TreeNode) -> str:
        counts = ", ".join(
            f"{label}:{count}" for label, count in sorted(node.class_counts.items())
        )
        if node.is_leaf:
            return (
                f"Leaf #{node.node_id}\n"
                f"class = {node.prediction}\n"
                f"gini = {node.gini:.2f}\n"
                f"samples = {node.num_samples}\n"
                f"counts = {counts}"
            )
        return (
            f"Node #{node.node_id}\n"
            f"{FEATURE_NAMES[node.feature_index]} <= {node.threshold:.2f}\n"
            f"gini = {node.gini:.2f}\n"
            f"samples = {node.num_samples}\n"
            f"class = {node.prediction}"
        )

    def draw_edges(node: TreeNode) -> None:
        x0, y0 = normalized_position(node)
        for child, label in [(node.left, "yes"), (node.right, "no")]:
            if child is None:
                continue
            x1, y1 = normalized_position(child)
            ax.plot([x0, x1], [y0, y1], color="black", linewidth=1.2)
            ax.text(
                (x0 + x1) / 2.0,
                (y0 + y1) / 2.0,
                label,
                fontsize=8,
                ha="center",
                va="center",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )
            draw_edges(child)

    def draw_nodes(node: TreeNode) -> None:
        x, y = normalized_position(node)
        facecolor = "#c8e6c9" if node.is_leaf else "#bbdefb"
        ax.text(
            x,
            y,
            node_text(node),
            ha="center",
            va="center",
            fontsize=7,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": facecolor,
                "edgecolor": "black",
                "alpha": 0.95,
            },
        )
        if node.left is not None:
            draw_nodes(node.left)
        if node.right is not None:
            draw_nodes(node.right)

    draw_edges(tree.root)
    draw_nodes(tree.root)
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def plot_voting_bars(
    ax: plt.Axes,
    demo: DemoState,
    votes_so_far: np.ndarray,
    vote_frame: int,
) -> None:
    if len(votes_so_far) == 0:
        vote_count_0 = 0
        vote_count_1 = 0
    else:
        counts_so_far = np.bincount(votes_so_far, minlength=2)
        vote_count_0 = int(counts_so_far[0])
        vote_count_1 = int(counts_so_far[1])

    ax.bar(["Class 0", "Class 1"], [vote_count_0, vote_count_1])
    ax.set_ylim(0, demo.n_trees)
    ax.set_ylabel("Number of Votes")
    ax.set_title("Voting Process")

    ax.text(
        0.5,
        demo.n_trees * 0.82,
        f"Votes so far:\nClass 0: {vote_count_0}\nClass 1: {vote_count_1}",
        ha="center",
        va="center",
        fontsize=12,
    )

    if vote_frame > demo.n_trees:
        ax.text(
            0.5,
            demo.n_trees * 0.45,
            f"Final Prediction: Class {demo.final_prediction}\n"
            f"P(Class 0) = {demo.prob_class_0:.2f}\n"
            f"P(Class 1) = {demo.prob_class_1:.2f}",
            ha="center",
            va="center",
            fontsize=13,
        )


def render_animation_frame(demo: DemoState, frame: int, fig: Optional[plt.Figure] = None) -> plt.Figure:
    """Render one frame of the full-process animation."""
    if fig is None:
        fig = plt.figure(figsize=(15, 6))
    else:
        fig.clear()

    frame = int(np.clip(frame, 0, demo.total_frames - 1))

    if frame < demo.phase_1_frames:
        depth = frame + 1
        current_tree = demo.tree1_build_models[frame]

        ax_tree = fig.add_subplot(1, 2, 1)
        ax_boundary = fig.add_subplot(1, 2, 2)

        draw_tree_structure(
            ax_tree,
            current_tree,
            f"Tree 1 Building Process: max_depth = {depth}",
        )
        draw_boundary(
            ax_boundary,
            demo,
            current_tree,
            f"Tree 1 Decision Boundary, depth = {depth}",
        )
        fig.suptitle("Stage 1: Building Tree 1", fontsize=15)

    elif frame < demo.phase_1_frames + demo.phase_2_frames:
        tree_idx = frame - demo.phase_1_frames
        current_tree = demo.forest.trees[tree_idx]

        ax_tree = fig.add_subplot(1, 2, 1)
        ax_boundary = fig.add_subplot(1, 2, 2)

        draw_tree_structure(
            ax_tree,
            current_tree,
            f"Tree Structure: Tree {tree_idx + 1}",
        )
        draw_boundary(
            ax_boundary,
            demo,
            current_tree,
            f"Decision Boundary of Tree {tree_idx + 1}",
        )
        fig.suptitle("Stage 2: Showing All Trees in the Forest", fontsize=15)

    else:
        vote_frame = frame - demo.phase_1_frames - demo.phase_2_frames

        ax_boundary = fig.add_subplot(1, 2, 1)
        ax_vote = fig.add_subplot(1, 2, 2)

        if vote_frame == 0:
            votes_so_far = np.array([], dtype=int)
            current_title = "New Sample Before Voting"
            draw_boundary(
                ax_boundary,
                demo,
                demo.forest,
                "Final Random Forest Boundary",
            )
        elif 1 <= vote_frame <= demo.n_trees:
            current_tree_idx = vote_frame - 1
            current_tree = demo.forest.trees[current_tree_idx]
            votes_so_far = demo.tree_votes_for_new_sample[:vote_frame]
            current_title = (
                f"Tree {current_tree_idx + 1} predicts: "
                f"Class {demo.tree_votes_for_new_sample[current_tree_idx]}"
            )
            draw_boundary(
                ax_boundary,
                demo,
                current_tree,
                f"Boundary of Voting Tree {current_tree_idx + 1}",
            )
        else:
            votes_so_far = demo.tree_votes_for_new_sample
            current_title = f"Final RF Prediction: Class {demo.final_prediction}"
            draw_boundary(
                ax_boundary,
                demo,
                demo.forest,
                "Final Random Forest Boundary",
            )

        ax_boundary.scatter(
            demo.new_sample[0, 0],
            demo.new_sample[0, 1],
            marker="*",
            s=350,
            edgecolor="k",
            label="New sample",
            zorder=10,
        )
        ax_boundary.legend(loc="upper right")
        ax_boundary.set_title(current_title)

        plot_voting_bars(ax_vote, demo, votes_so_far, vote_frame)
        fig.suptitle("Stage 3: Voting and Final Output", fontsize=15)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def save_animation_gif(demo: DemoState, output_path: Union[str, Path], fps: int = 1) -> Path:
    """Save the complete Random Forest process animation as a GIF."""
    output_path = Path(output_path)
    fig = plt.figure(figsize=(15, 6))

    def update(frame: int) -> None:
        render_animation_frame(demo, frame, fig=fig)

    animation = FuncAnimation(
        fig,
        update,
        frames=demo.total_frames,
        interval=1200,
        repeat=True,
    )
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output_path


def format_terminal_output(demo: DemoState) -> str:
    """Return a concise Chinese voting summary."""
    lines = ["每棵树的预测结果："]
    for i, pred in enumerate(demo.tree_votes_for_new_sample, start=1):
        lines.append(f"第 {i} 棵树：类别 {pred}")

    lines.extend(
        [
            "",
            "投票统计：",
            f"类别 0：{demo.final_vote_counts[0]} 票",
            f"类别 1：{demo.final_vote_counts[1]} 票",
            "",
            "随机森林最终预测：",
            f"类别 {demo.final_prediction}",
            "",
            "估计类别概率：",
            f"P(类别 0) = {demo.prob_class_0:.2f}",
            f"P(类别 1) = {demo.prob_class_1:.2f}",
        ]
    )
    return "\n".join(lines)


# ============================================================
# 4. Streamlit 页面
# ============================================================


def initialize_session_state() -> None:
    defaults = {
        "demo": None,
        "gif_bytes": None,
        "gif_name": "rf_classification_full_process.gif",
        "current_frame": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main() -> None:
    st.set_page_config(
        page_title="自实现随机森林可视化器",
        layout="wide",
    )
    initialize_session_state()

    st.title("自实现 Random Forest 随机森林可视化")
    st.write(
        "本页面展示一个从零实现的 Random Forest 分类器："
    )

    with st.sidebar:
        st.header("参数设置")

        n_samples = st.slider("样本数量", 80, 500, 260, 20)
        noise = st.slider("数据噪声", 0.0, 0.6, 0.25, 0.05)
        data_seed = st.number_input("数据随机种子", 0, 9999, 42, 1)

        st.divider()
        n_trees = st.slider("树的数量", 1, 15, 9, 2)
        max_depth = st.slider("每棵树最大深度", 1, 5, 3, 1)
        min_samples_split = st.slider("节点继续分裂所需最少样本数", 2, 20, 2, 1)
        max_features = st.selectbox(
            "每次分裂随机考虑的特征数",
            options=[1, 2, None, "sqrt", "log2"],
            index=0,
            format_func=lambda value: "全部特征" if value is None else str(value),
        )
        forest_seed = st.number_input("森林随机种子", 0, 9999, 7, 1)

        st.divider()
        st.subheader("待预测新样本")
        new_sample_x = st.number_input("Feature 1", -3.0, 4.0, 0.25, 0.05)
        new_sample_y = st.number_input("Feature 2", -2.5, 2.5, 0.35, 0.05)
        grid_resolution = st.slider("分类边界网格精度", 80, 300, 180, 20)

        st.divider()
        train_clicked = st.button("训练并生成逐帧动画", use_container_width=True)
        if train_clicked:
            st.session_state.demo = build_demo_state(
                n_samples=int(n_samples),
                noise=float(noise),
                data_seed=int(data_seed),
                n_trees=int(n_trees),
                max_depth=int(max_depth),
                min_samples_split=int(min_samples_split),
                max_features=max_features,
                forest_seed=int(forest_seed),
                new_sample_x=float(new_sample_x),
                new_sample_y=float(new_sample_y),
                grid_resolution=int(grid_resolution),
            )
            st.session_state.gif_bytes = None
            st.session_state.current_frame = 0

        if st.button("重置", use_container_width=True):
            st.session_state.demo = None
            st.session_state.gif_bytes = None
            st.session_state.current_frame = 0

    tab_principle, tab_animation = st.tabs(["RF 原理", "逐帧动画"])

    with tab_principle:
        st.subheader("Random Forest 的核心思想")
        st.markdown(
            """
            Random Forest（随机森林）是由很多棵 Decision Tree 组成的分类模型。它的关键不是训练一棵最强的树，而是训练多棵彼此有差异的树，然后让它们共同投票。

            **1. Bootstrap 有放回采样**  
            训练第 k 棵树时，不直接使用完整训练集，而是从原始训练集中随机抽样。抽样是“有放回”的，所以同一个样本可能被抽到多次，也可能完全没被抽到。这样每棵树看到的数据都不同。

            **2. 节点随机特征选择**  
            每棵树在某个节点寻找最佳分裂时，不一定查看所有特征，而是随机选一部分特征作为候选。这样能进一步增加树和树之间的差异。

            **3. Gini impurity 衡量节点混乱程度**  
            $Gini = 1 - sum(p_k^2)$。其中 p_k 表示当前节点里第 k 类样本的比例。Gini 越小，说明节点越纯；如果一个节点全是同一类样本，Gini 就是 0。

            **4. 贪心选择最佳划分**  
            决策树会枚举候选特征和候选阈值，把样本分到左右子节点，并选择加权 Gini 最小的划分。这个过程递归进行，直到达到最大深度、样本太少、节点已经纯净，或找不到有效划分。

            **5. 多数投票得到最终预测**  
            预测一个新样本时，每棵树都会给出一个类别。随机森林统计所有树的投票数，选择票数最多的类别作为最终预测。如果平票，本实现选择标签值较小的类别。
            """
        )

    with tab_animation:
        demo: Optional[DemoState] = st.session_state.demo
        if demo is None:
            st.info("请先在左侧点击 **训练并生成逐帧动画**。")
            return

        st.success(
            f"模型已训练完成：{demo.n_trees} 棵树，最大深度 {demo.max_depth}，"
            f"动画共 {demo.total_frames} 帧。"
        )

        st.session_state.current_frame = max(
            0,
            min(int(st.session_state.current_frame), demo.total_frames - 1),
        )

        left_button, frame_info, right_button = st.columns([1, 2, 1])

        with left_button:
            if st.button(
                "⬅️ 上一帧",
                use_container_width=True,
                disabled=st.session_state.current_frame <= 0,
            ):
                st.session_state.current_frame -= 1
                st.rerun()

        with frame_info:
            st.markdown(
                f"<div style='text-align:center; font-size:18px;'>"
                f"当前帧：{st.session_state.current_frame + 1} / {demo.total_frames}"
                f"</div>",
                unsafe_allow_html=True,
            )

        with right_button:
            if st.button(
                "下一帧 ➡️",
                use_container_width=True,
                disabled=st.session_state.current_frame >= demo.total_frames - 1,
            ):
                st.session_state.current_frame += 1
                st.rerun()

        current_figure = render_animation_frame(
            demo,
            st.session_state.current_frame,
        )
        st.pyplot(current_figure)
        plt.close(current_figure)

        st.subheader("最终投票结果")
        st.code(format_terminal_output(demo), language="text")

if __name__ == "__main__":
    main()
