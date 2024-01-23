class Node:
    def __init__(self, node_id: int = None, thought: str = None, action: str = None, result: str = None,
                 score: float = None, hint: str = ''):
        self.id = node_id if node_id is not None else self._next_id()
        self.thought = thought
        self.action = action
        self.result = result
        self.score = score
        self.hint = hint
        self.parent = None
        self.children = []
        self.depth = 0
        self.is_leaf = False

    _next_id_counter = 1

    @classmethod
    def _next_id(cls):
        node_id = cls._next_id_counter
        cls._next_id_counter += 1
        return f'node_{node_id}'

    def add_parent(self, parent_node):
        self.parent = parent_node
        self.depth = self.parent.depth + 1

    def add_child(self, child_node):
        self.children.append(child_node)

    def as_dict(self):
        return {
            "Thought": self.thought,
            "Action": self.action,
            "Result": self.result,
        }

    def as_string(self):
        return f"""Thought: {self.thought}\nAction: {self.action}\nResult: {self.result}"""

    def __repr__(self):
        return f"Node(id={self.id} children={len(self.children)} depth:{self.depth} is_leaf={self.is_leaf})"

