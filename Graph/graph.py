import os
import networkx as nx
from pyvis.network import Network


class Graph:
    def __init__(self, name='Visualizations/GoAT.html', height='1000px', width='100%', bg_color='white',
                 font_color='black',
                 directed=True, pruning_threshold=0.5):
        self.name = name
        self.graph = nx.DiGraph() if directed else nx.Graph()
        self.height = height
        self.width = width
        self.bg_color = bg_color
        self.font_color = font_color

        self.pruning_threshold = pruning_threshold
        self.node_properties = {}

        self.color_map = []

    def _generate_color(self, valid=True):
        color = '#49d159' if valid else '#d14949'
        return color

    def add_node(self, node, node_type='Object', is_init=False):
        size = 15
        color = "#429bf5" if node_type == 'Object' else "#9eebfa" if is_init else self._generate_color(valid=True)
        shape = 'dot' if node_type == 'Object' else 'box'

        if is_init:
            pass
        else:
            self.graph.add_node(node.id,
                                node_obj=node.as_dict(),
                                depth=node.depth,
                                label=node.id,
                                color=color,
                                size=size,
                                title=node.as_string() + f"\n\nScore: {node.score}",
                                shape=shape,
                                borderWidth=2
                                )
            # Add edges to the NetworkX graph
            if node.parent is not None:
                self.graph.add_edge(node.parent.id, node.id)

    def add_edge(self, edge, label=None, title=None):
        self.graph.add_edge(*edge, label=label, title=title)

    def get_nodes(self):
        return [node for node, data in self.node_properties.items()]

    def highlight_solution(self, solution):
        for id, data in self.graph.nodes.items():
            if id in [n.id for n in solution]:
                data["color"] = "#944ef5"
            else:
                data["color"] = "#429bf5"

    def show_graph(self, show_buttons=False, layout_options=False):
        pyvis_graph = Network(height=self.height, width=self.width, bgcolor=self.bg_color, font_color=self.font_color,
                              directed=self.graph.is_directed())
        pyvis_graph.from_nx(self.graph)

        if show_buttons:
            pyvis_graph.show_buttons(filter_=['physics', 'layout'])

        if layout_options:
            const_options = """{
                "layout": {
                    "hierarchical": {
                        "enabled": true,
                        "levelSeparation": 120,
                        "nodeSpacing": 200,
                        "treeSpacing": 150,
                        "direction": "UD",
                        "sortMethod": "directed",
                        "shakeTowards": "roots"
                    }
                },
                "physics": {
                    "hierarchicalRepulsion": {
                        "centralGravity": 0,
                        "springLength": 120,
                        "springConstant": 0,
                        "avoidOverlap": 0.1
                    },
                    "minVelocity": 0.75,
                    "solver": "hierarchicalRepulsion"
                }
            }"""
            pyvis_graph.set_options(const_options)

        # Create the base directory if it doesn't exist
        base_dir = self.name.split('/')[0]
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        pyvis_graph.save_graph(self.name)
