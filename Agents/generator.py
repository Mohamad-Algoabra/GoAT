from langchain_core.messages import HumanMessage

from Agents.agent import Agent
from Agents.LLM import LLM
from Utils.utils import count_tokens


class Generator(Agent):
    """
    The Generator class is responsible for the creation and generation of new thoughts based on given inputs.
    """

    def __init__(self, **model_parameters):
        super().__init__()
        self.model = LLM(**model_parameters).get_model()

        self.system_prompt = (
            "You are an smart expert in {domain}."
            "Your role involves creating step-by-step solutions for complex problems."
            "The given problem:\n"
            "{initial_prompt}\n\n"
            "Decompose the problem into a strategic plan with clear, actionable short steps, "
            "aiming for the least number of steps possible to solve this problem. "
            "Each step should include a concise short thought, action and its expected result. "
            "Format your response as structured compilation to the current reasoning states,"
            "focusing on concise and abstract thinking step-by-step:\n\n"
            "\n'Step': The number of the step:\n"
            "\n'Thought': A short, abstract smart description of what should be done, derived from the preceding step,"
            "\n'Action': A straightforward action applying the thought,"
            "\n'Result': A concise description of the intermediate or final result post-action"
            "\n"
        )

        self.task_prompt = (
            "So far, we have considered the following approaches, but they haven't worked out:\n"
            "{rejected_actions}\n\n"
            "Now, let's explore why they do not work and make better decisions. "
            "Consider the current situation and the factors at play:\n"
            "{reasoning_states}\n\n"
            "Based on this, what better decisions (correct actions) or different short thoughts you we think of?\n\n"
            "{evaluator_hint}" + "\nThink with clear thoughts and correct math and do not forget details.\n"
            "We are currently at Step {step_number}. "
            "What should our next moves be? be concise and clear\n"
            "\nStep {step_number}.:\n"
        )
        self.tokens_count = 0

    def generate(self, initial_prompt: str, domain: str, reasoning_states: str, rejected_actions: str, hint: str,
                 step_number: int) -> str:
        """
        Generates response from the input data.
        """
        print("\n=====> Starting Generating <=====")

        prompt = self._generate_model_prompt(system_prompt=self.system_prompt,
                                             task_prompt=self.task_prompt,
                                             input_variables=["initial_prompt",
                                                              "domain",
                                                              "reasoning_states",
                                                              "rejected_actions",
                                                              "step_number",
                                                              "evaluator_hint"])

        message = prompt.format_messages(initial_prompt=initial_prompt,
                                         domain=domain,
                                         reasoning_states=reasoning_states,
                                         rejected_actions=rejected_actions,
                                         step_number=step_number,
                                         evaluator_hint=hint)
        # print('Prompt', '-' * 50)
        # print(message[1].content)

        result = self.model(message).content

        self.tokens_count += count_tokens(text=result)

        return result

    def generate_solution(self, init_problem, path):
        answer_path = [node.as_string() for node in path]
        answer_path = '\n'.join(answer_path)

        task_prompt = (
            "Based on the following steps taken to solve this problem:"
            "{init_problem}\n\n"
            "Steps:\n"
            "{answer_path}\n\n"
            "Considering the thoughts, actions and results outlined in these steps, "
            "we need to synthesize our findings and insights. "
            "What conclusion or solution do they lead us to?\n\n"
            "Output your answer and do not return any thing else:\n"
            "{{'Final Answer': answer}}"
            "let's determine the final answer:\n\n"
        )

        prompt = self._generate_model_prompt(system_prompt=" ", task_prompt=task_prompt,
                                             input_variables=["init_problem", "answer_path"])
        message = prompt.format_messages(init_problem=init_problem, answer_path=answer_path)

        result = self.model(message).content
        self.tokens_count += count_tokens(text=result)

        return result

