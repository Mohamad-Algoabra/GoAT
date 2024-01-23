from abc import ABC, abstractmethod

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


class Agent(ABC):
    """
    An abstract base class that defines the structure and required methods
    for agents that process inputs and generate outputs.
    """

    @abstractmethod
    def __init__(self):
        """
        Initialize the agent with necessary components.
        """
        pass

    @staticmethod
    def _generate_model_prompt(system_prompt: str, task_prompt: str, input_variables: list) -> ChatPromptTemplate:
        """
        Generates a prompt for the model using the provided system prompt.

        :param system_prompt: The prompt defining the task and expected output format.
        :return: A formatted ChatPromptTemplate object.
        """

        return ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(system_prompt),
                HumanMessagePromptTemplate.from_template(task_prompt)
            ],
            input_variables=input_variables
        )

