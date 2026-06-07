from dataclasses import dataclass, field


# 所有允许的可视化事件类型。
# 使用常量可以避免在其他文件里反复手写字符串，减少拼写错误。
BOOTSTRAP_START = "bootstrap_start"
BOOTSTRAP_SAMPLE_SELECTED = "bootstrap_sample_selected"
TREE_TRAINING_START = "tree_training_start"
NODE_SPLIT_START = "node_split_start"
SPLIT_CHOSEN = "split_chosen"
LEAF_CREATED = "leaf_created"
TREE_TRAINING_END = "tree_training_end"
PREDICTION_TREE_START = "prediction_tree_start"
PREDICTION_PATH_STEP = "prediction_path_step"
TREE_VOTE = "tree_vote"
FOREST_VOTE_UPDATE = "forest_vote_update"
PREDICTION_END = "prediction_end"


VALID_EVENT_TYPES = {
    BOOTSTRAP_START,
    BOOTSTRAP_SAMPLE_SELECTED,
    TREE_TRAINING_START,
    NODE_SPLIT_START,
    SPLIT_CHOSEN,
    LEAF_CREATED,
    TREE_TRAINING_END,
    PREDICTION_TREE_START,
    PREDICTION_PATH_STEP,
    TREE_VOTE,
    FOREST_VOTE_UPDATE,
    PREDICTION_END,
}


@dataclass
class VisualizationEvent:
    """
    A single visualization event.

    这个类只负责保存“算法运行到了哪一步”。
    它不负责训练模型，也不负责画图。

    后续 plot_utils.py 和 app.py 可以读取这些事件，
    然后一步一步展示 Bootstrap、建树、预测路径和投票过程。
    """

    # 事件类型，例如 "bootstrap_start" 或 "split_chosen"。
    event_type: str

    # 当前事件属于第几棵树。
    # 如果这个事件不是某一棵树专属的，可以设为 None。
    tree_index: int | None = None

    # 当前事件属于哪个树节点。
    # 例如节点分裂、创建叶子节点时会用到。
    # 如果事件和节点无关，可以设为 None。
    node_id: int | None = None

    # 给用户看的简短说明文字。
    message: str = ""

    # 不同事件需要保存的额外信息不完全一样。
    # 例如：
    # - bootstrap 事件可以保存 sample_indices
    # - split 事件可以保存 feature_index、threshold、gini
    # - vote 事件可以保存 vote_counts
    data: dict = field(default_factory=dict)

    def __post_init__(self):
        """
        dataclass 创建对象后会自动调用这个方法。

        这里用来检查 event_type 是否合法。
        如果事件类型拼错，程序会尽早报错，方便调试。
        """
        if self.event_type not in VALID_EVENT_TYPES:
            raise ValueError(f"Unknown visualization event type: {self.event_type}")

    def to_dict(self):
        """
        Convert the event to a dictionary.

        Streamlit 和 Matplotlib 处理字典会比较方便，
        所以提供这个方法，方便之后的可视化模块直接使用。
        """
        return {
            "event_type": self.event_type,
            "tree_index": self.tree_index,
            "node_id": self.node_id,
            "message": self.message,
            "data": self.data,
        }


class EventRecorder:
    """
    A simple recorder for visualization events.

    算法模块训练或预测时，可以调用 record() 记录事件。
    可视化模块不需要知道算法细节，只需要按顺序读取 events。
    """

    def __init__(self):
        # 按算法执行顺序保存所有事件。
        self.events = []

    def record(
        self,
        event_type,
        tree_index=None,
        node_id=None,
        message="",
        **data,
    ):
        """
        Create and save a visualization event.

        Parameters
        ----------
        event_type : str
            事件类型，必须属于 VALID_EVENT_TYPES。

        tree_index : int or None
            当前事件对应的树编号。

        node_id : int or None
            当前事件对应的节点编号。

        message : str
            给用户看的解释文字。

        **data
            额外信息，例如 sample_indices、feature_index、threshold、gini 等。

        Returns
        -------
        VisualizationEvent
            刚刚创建并保存的事件对象。
        """
        event = VisualizationEvent(
            event_type=event_type,
            tree_index=tree_index,
            node_id=node_id,
            message=message,
            data=data,
        )

        self.events.append(event)
        return event

    def get_events(self):
        """
        Return all events in order.

        返回的是事件对象列表，不是字典列表。
        如果 app.py 需要字典，可以调用 get_event_dicts()。
        """
        return self.events

    def get_event_dicts(self):
        """
        Return all events as dictionaries.

        这个方法适合给 Streamlit 或绘图函数使用。
        """
        return [event.to_dict() for event in self.events]

    def get_events_by_type(self, event_type):
        """
        Return events with a specific event type.

        例如只想查看所有 split_chosen 事件时可以使用。
        """
        if event_type not in VALID_EVENT_TYPES:
            raise ValueError(f"Unknown visualization event type: {event_type}")

        return [
            event
            for event in self.events
            if event.event_type == event_type
        ]

    def clear(self):
        """
        Remove all recorded events.

        重新训练模型或点击 Reset 时可以调用。
        """
        self.events.clear()
