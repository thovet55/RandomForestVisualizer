from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class TreeNode:
    node_id: int

    # Depth of this node in the tree. Root node has depth 0.
    depth: int

    # Index of the feature used for splitting at this node.
    # For example, feature_index = 0 means using X[:, 0].
    # Leaf nodes do not split, so this can be None.
    feature_index: Optional[int] = None

    # Threshold value used for splitting.
    # Samples with feature value <= threshold go left,
    # and samples with feature value > threshold go right.
    threshold: Optional[float] = None

    # Left child node.
    left: Optional["TreeNode"] = None

    # Right child node.
    right: Optional["TreeNode"] = None

    # Predicted class at this node.
    # For leaf nodes, this is the final prediction.
    prediction: Optional[Any] = None

    # Gini impurity of the samples at this node.
    gini: Optional[float] = None

    # Number of training samples that reach this node.
    num_samples: int = 0

    # Count of each class among samples at this node.
    # Example: {0: 5, 1: 3}
    class_counts: dict = field(default_factory=dict)

    # Whether this node is a leaf node.
    is_leaf: bool = False
