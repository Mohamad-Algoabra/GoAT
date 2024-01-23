from Agents.agent import Agent
from Agents.LLM import LLM
from Utils.utils import count_tokens


class Evaluator(Agent):
    """
    The Evaluator class is responsible for the evaluation and guiding of new thoughts based on given inputs.
    """

    def __init__(self, **model_parameters):
        super().__init__()
        self.model = LLM(**model_parameters).get_model()

        self.system_prompt = (
            "You are an expert in {domain}, designed to evaluate the effectiveness of the given solution step for the given problem. "
            "Your task is to provide a score ranging from 0 (ineffective) to 100 (highly effective). "
            "Assess the solution step based on its logical coherence, alignment with problem requirements, "
            "and its overall impact on the solution.\n\n"
            "Sometimes it may be a correct step, but it misses a step before it, so the final answer is wrong."
            "or it maybe a correct thought but false result"
            "Output your evaluation as follow:\n"
            "\n'Final Score': score."
            "\n'Hint': Rethink about the step and briefly suggest a more effective step or correction (recalculation) "
            "and if the step correct, confirm it!."
            "Consider that the step might be an intermediate one, building upon previous steps. "
            "Let your output be as short as possible!"
        )

        self.task_prompt = (
            "Problem:\n{initial_prompt}\n\n"
            "Previous steps:\n{reasoning_states}\n\n"
            "Evaluate this step:\n{state}\n\n"
            "Determine the effectiveness of this step in solving the given problem. "
            "Provide a score and a brief hint for improvement if necessary."
        )

        self.tokens_count = 0

    def evaluate(self, input_data, thought, domain, reasoning_states):
        """
        Evaluate thoughts from the input data.

        :return: An evaluation of the thought or hypothesis.
        """
        print("\n=====> Starting Evaluating <=====")

        prompt = self._generate_model_prompt(system_prompt=self.system_prompt,
                                             task_prompt=self.task_prompt,
                                             input_variables=["initial_prompt", "domain", "reasoning_states", "state"])

        message = prompt.format_messages(initial_prompt=input_data, state=thought, domain=domain,
                                         reasoning_states=reasoning_states)

        result = self.model(message).content

        self.tokens_count += count_tokens(text=result)

        return result

    def evaluate_path(self, node_path):
        node_sum = 0
        for node in node_path[1:]:
            node_sum += node.score

        path_score = node_sum / (len(node_path) - 1)
        return path_score
