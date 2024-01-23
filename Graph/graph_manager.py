from queue import PriorityQueue
from typing import List, Any, Optional, Callable, Dict, Tuple
from Agents.parser import Parser
from Agents.generator import Generator
from Agents.evaluator import Evaluator

from Graph.graph import Graph
from Graph.node import Node
from Prompts.prompts import *


class GraphManager:
    """
    Manages the operations on the thought graph.
    It integrates a generator, evaluator and parser agents to manage and optimize the thought process.
    """

    def __init__(self, initial_prompt: str, generator: Generator, evaluator: Evaluator, parser: Parser,
                 node_threshold: float, path_threshold: float, max_width: int, max_depth: int):
        """
        Initializes the GraphManager.

        :param initial_prompt: The initial problem to start the graph.
        :param generator: The Generator instance for generating new thoughts.
        :param evaluator: The Evaluator instance for evaluating the thoughts.
        :param parser: The Parser instance for parse the outputs.
        :param node_threshold: A threshold score for pruning less promising nodes. Nodes with scores below this threshold are not further expanded.
        :param path_threshold: A threshold score for validating the solution paths. Paths with scores above this threshold are considered valid solutions and stops the search.
        :param max_width: The maximum number of child nodes each node in the graph can have, controlling the breadth of exploration.
        :param max_depth: The maximum depth the graph can expand to, controlling the depth of exploration.
        """
        self.initial_prompt = initial_prompt.strip()

        self.parser = parser
        self.generator = generator
        self.evaluator = evaluator

        self.max_width = max_width
        self.max_expand_depth = max_depth
        self.score_threshold = node_threshold
        self.path_threshold = path_threshold

        self.parsed_data = self.parser.parse(data=self.initial_prompt,
                                             output_format=output_formats['input_format'],
                                             expected_keys=output_formats['input_expected_keys'])[0]

        self.root_node = Node(thought=self.parsed_data['Prior_Knowledge'],
                              action=self.parsed_data['Question'])
        self.graph = Graph()
        self.graph.add_node(self.root_node)

        self.graph_dict = {self.root_node.id: self.root_node}

        # self.visited = set()
        self.visited = []

        self.algorithms: Dict[str, Callable[..., Optional[List[Node]]]] = {
            'search': self.search
            # Other algorithms can be added here.
        }

        self.potential_solutions = []

        self.tokens_count = 0
        self.final_answer = None

    def expand_node(self, node: Node):
        """
        Central function that coordinates interactions between the generator, parser, and evaluator agents to expand a given node.
        It generates a new chain of nodes from the current node, structures the output, and evaluates each new node's relevance and quality.

        :param node: The node to be expanded.
        """

        print(f'\n\n=====> Expanding {node} <=====')

        # Create the state (reasoning path) for the generator
        state, state_number, path = self.create_reasoning_path(node)

        # Convert the visited solutions into a format suitable for the generator
        frozen_state = {frozenset(d.items()) for d in path}
        filtered_list = [f"- {d['Result']}\n" for d in self.visited if
                         frozenset(d.items()) not in frozen_state]
        visited_states_str = '\n'.join(
            node for node in filtered_list) if filtered_list else 'There is no observations yet!'

        print('-' * 100)
        print('\n', visited_states_str, '\n')
        print('-' * 100)
        print('\n', state, '\n')
        print('-' * 100)

        # Generate new chain considering the state and rejected states
        generated_chain = self.generator.generate(
            initial_prompt=self.initial_prompt,
            domain=self.parsed_data['Domain'],
            reasoning_states=state,
            rejected_actions=visited_states_str,
            step_number=state_number + 1,
            hint=f'Hint: {node.hint}' if node.hint else ''
        )

        generated_chain = self.parser.filter_duplicate_thoughts(
            self.parser.parse_output(text=generated_chain,
                                     output_format=output_formats['thoughts_format'],
                                     keys=output_formats['thoughts_expected_keys'])
        )

        # Keep track of the last node in the chain
        node.is_leaf = False
        parent_node = node
        loop_completed = True  # Flag to track if the loop completes without a break

        for thought in generated_chain:
            child_node = Node(thought=thought['Thought'], action=thought['Action'], result=thought['Result'])

            child_state, _, _ = self.create_reasoning_path(node)

            child_node_eval = self.evaluator.evaluate(input_data=self.initial_prompt, thought=child_node.as_string(),
                                                      domain=self.parsed_data['Domain'],
                                                      reasoning_states=child_state)

            parsed_eval = self.parser.parse_output(text=child_node_eval,
                                                   output_format=output_formats['evaluation_format'],
                                                   keys=output_formats['evaluation_expected_keys'])

            # print('\nDebugging:', parsed_eval, type(parsed_eval))
            score = float(parsed_eval[0]['Final Score']) / 100
            child_node.score = score
            parent_node.hint = parsed_eval[0]['Hint']

            self.visited.append(child_node.as_dict())
            # Add the child node only if it meets the score threshold
            if child_node.score >= self.score_threshold:
                child_node.add_parent(parent_node)
                parent_node.add_child(child_node)

                self.graph.add_node(child_node)

                self.graph_dict[parent_node.id] = parent_node  # .children.append(child_node)
                self.graph_dict[child_node.id] = child_node  # []

                parent_node = child_node  # Update the last node in the chain
            else:
                # self.rejected_solutions.append(child_node.as_string())
                loop_completed = False  # Set the flag to False as the loop breaks here
                break  # Stop processing further nodes if a node is below the threshold

        # Set the is_leaf attribute only if the loop was completed
        if loop_completed:
            parent_node.is_leaf = True

    def create_reasoning_path(self, node: Node) -> tuple[str, int, any]:
        """
        Creates a string representation of the reasoning path leading up to the given node.
        """
        # Traverse back to the root to construct the reasoning path
        path = []
        current_node = node
        while current_node:
            if current_node.parent is None:
                state = {'Initial State': self.parsed_data['Prior_Knowledge']}
            else:
                state = current_node.as_dict()
            path.append(state)
            current_node = current_node.parent

        # Add indices to the path elements

        # path_with_indices = [f"Step {i}. :\n{state}" for i, state in enumerate(reversed(path), start=1)]
        path_with_indices = [f"Step {i}:\n{' '.join([f'{k}: {v}' for k, v in state.items()])}"
                             for i, state in enumerate(reversed(path), start=1)]

        return '\n'.join(path_with_indices), len(path_with_indices), path

    def solve(self, search_algorithm: str, *args, **kwargs) -> Optional[List[Any]]:
        """
        Solves the problem by exploring the thought graph using a specified search algorithm.

        Args:
            search_algorithm: The name of the search algorithm to use.
            *args: Positional arguments passed to the search algorithm.
            **kwargs: Keyword arguments passed to the search algorithm.

        Returns:
            A list of Nodes representing the solution path, if found.
        """
        algorithm = self.algorithms.get(search_algorithm)
        if algorithm:
            self.expand_node(self.root_node)
            optimal_solution = algorithm(*args, **kwargs)
            final_answer = self.generator.generate_solution(init_problem=self.initial_prompt, path=optimal_solution)

            self.tokens_count = self.generator.tokens_count + self.evaluator.tokens_count + self.parser.tokens_count

            return final_answer
        else:
            raise ValueError(f"Search algorithm '{search_algorithm}' is not supported.")

    def search(self, iteration_limit=50, down_up=True):
        """
        Searches the graph using a priority queue-based approach to find the solution.
        """

        # Initialize a priority queue for nodes and a set to track enqueued nodes
        node_queue = PriorityQueue()
        enqueued_nodes = set()

        def enqueue_nodes():
            for node_id, node_data in self.graph.graph.nodes(data=True):
                # if the node still can be expanded
                if node_id not in enqueued_nodes:
                    priority = -node_data['depth'] if down_up else node_data['depth']
                    node_queue.put((priority, node_id))
                # if the node reached the max children number
                if len(self.graph_dict[node_id].children) >= self.max_width:
                    enqueued_nodes.add(node_id)

        # Initial enqueue of expandable nodes
        enqueue_nodes()

        # Helper function to enqueue nodes with certain conditions
        # def enqueue_nodes():
        #     for node_id, node_data in self.graph.graph.nodes(data=True):
        #         if node_data['depth'] <= self.max_expand_depth and node_id not in enqueued_nodes:
        #             priority = -node_data['depth'] if down_up else node_data['depth']
        #             node_queue.put((priority, node_id))
        #             if len(self.graph_dict[node_id].children) >= self.max_width:
        #                 enqueued_nodes.add(node_id)

        for _ in range(iteration_limit):

            if node_queue.empty():
                break

            _, current_node_id = node_queue.get()
            # print('=' * 50, current_node_id)
            current_node = self.graph_dict[current_node_id]

            # Return solution path if a valid leaf node is found
            if current_node.is_leaf:
                potential_solution_path = self.create_solution_path(current_node)
                path_score = self.evaluator.evaluate_path(potential_solution_path)
                if path_score > self.path_threshold:
                    self.final_answer = potential_solution_path
                    self.graph.highlight_solution(self.final_answer)
                    return potential_solution_path
                else:
                    self.potential_solutions.append((potential_solution_path, path_score))

            elif current_node.depth <= self.max_expand_depth and len(current_node.children) < self.max_width:
                # Expand the current node if conditions are met and enqueue new nodes
                self.expand_node(current_node)
                enqueue_nodes()

        for i, n in self.graph_dict.items():
            if len(n.children) < 1:
                potential_solution_path = self.create_solution_path(n)
                path_score = self.evaluator.evaluate_path(potential_solution_path)
                self.potential_solutions.append((potential_solution_path, path_score))

        # If no valid solution path is found return the path with highest score.
        self.final_answer = max(self.potential_solutions, key=lambda x: x[1])[0]
        self.graph.highlight_solution(self.final_answer)
        return self.final_answer

    def create_solution_path(self, node: Node) -> list:
        """
        """
        path = []
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

