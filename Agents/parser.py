import re
from typing import List, Dict, Any, Union
from Utils.utils import extract_and_validate, count_tokens
from Agents.LLM import LLM
from Agents.agent import Agent
# from Prompts.prompts import parser_configs


class Parser(Agent):
    """
    The Parser class processes complex inputs into structured data that can be
    utilized by the Generator and Evaluator classes.
    """

    def __init__(self, **model_parameters):
        super().__init__()
        self.model = LLM(**model_parameters).get_model()

        self.system_prompt = (
            "Your role is to format text inputs into structured JSON.\n"
            "Format the output as a JSON object as follows:\n"
            "{output_format}"
        )

        self.task_prompt = "Parse this input:\n{input_data}\n\n"

        self.tokens_count = 0

    def parse(self, data: str, output_format: str, expected_keys: List[str]) -> \
            Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Parses input data into a structured JSON format using the model.

        :param data: The raw input data to parse.
        :param output_format:
        :param expected_keys: The keys expected in the parsed result.
        :return: The parsed result as a structured JSON.
        """

        print("\n=====> Starting Parsing <=====")
        prompt = self._generate_model_prompt(system_prompt=self.system_prompt,
                                             task_prompt=self.task_prompt,
                                             input_variables=["output_format", "input_data"])

        formatted_message = prompt.format_messages(output_format=output_format, input_data=data)
        result = self.model(formatted_message).content

        self.tokens_count += count_tokens(text=result)

        return self.parse_output(text=result, output_format=output_format, keys=expected_keys)

    def parse_output(self, text, output_format, keys):

        def validate_dicts(dict_list, required_keys):
            """
            This function takes a list of dictionaries and a set of required keys,
            and checks if all dictionaries in the list contain all the required keys.
            """
            if not dict_list:
                return False

            for d in dict_list:
                # Check if each dictionary contains all the required keys
                if not all(key in d for key in required_keys):
                    return False
                if 'Final Score' in d:
                    try:
                        # Extract the value associated with the key 'Final Score'
                        value_str = d['Final Score']

                        # Attempt to convert the extracted value to a float
                        _ = float(value_str)

                        return True
                    except ValueError:
                        # Return False if the conversion to float fails
                        return False
            return True

        def build_pattern(key):
            # Build a regular expression pattern for the given key
            if key == 'Final Score':
                return rf"[\"']?{key}[\"']?\s*:\s*(\d+)"
            else:
                return rf"[\"']?{key}[\"']?\s*:\s*(.*?)(?=\n*\s*[\"']?(?:{'|'.join(keys)})[\"']?|\n\n|}}|Step|$)"

        # Create a dictionary of patterns for each key
        patterns = {key: re.compile(build_pattern(key), re.DOTALL) for key in keys}

        parsed_data = []

        # Find all matches for each key
        matches = {key: re.findall(pattern, text) for key, pattern in patterns.items()}

        # Determine the maximum number of entries
        max_length = max(len(match) for match in matches.values())

        # Iterate over the matches and build the data structure
        for i in range(max_length):
            parsed_entry = {key: matches[key][i].strip() if i < len(matches[key]) else "N/A" for key in keys}
            parsed_data.append(parsed_entry)

        if validate_dicts(parsed_data, keys):
            return parsed_data
        else:
            print('<', '=' * 30, 'Reparse')
            return self.parse(data=text, output_format=output_format, expected_keys=keys)

    def filter_duplicate_thoughts(self, record_list):
        """
        Filters out duplicate records from a list of dictionaries based on the 'Thought' key.

        Args:
        record_list (list of dict): List containing dictionaries with 'Thought', 'Action', 'Result' keys.

        Returns:
        list of dict: Filtered list with duplicates removed.
        """
        seen_thoughts = set()
        filtered_list = []

        for record in record_list:
            # Normalize the 'Thought' string to ensure consistent comparison
            thought = record['Thought'].replace('"', '').replace("'", '').strip().lower()

            if thought not in seen_thoughts:
                seen_thoughts.add(thought)
                filtered_list.append(record)

        return filtered_list

    # def build_knowledge_graph(self, data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    #     """
    #     Constructs a knowledge graph from the provided data.
    #
    #     :param data: Data containing prior knowledge and domain information.
    #     :return: A dictionary containing entities and relationships of the knowledge graph.
    #     """
    #     entities = self.parse(
    #         data=data['Prior_Knowledge'],
    #         system_prompt=f"This task is related to {data['Domain']}" + parser_configs['system_entities_parser'],
    #         expected_keys=parser_configs['entities_expected_keys'],
    #         is_list=True
    #     )[0]
    #     entities_list = [item["Name"] for item in entities]
    #     relations = self.parse(
    #         data=f"{data['Prior_Knowledge']}\nEntities: {entities_list}",
    #         system_prompt=f"This task is related to {data['Domain']}" + parser_configs['system_relations_parser'],
    #         expected_keys=parser_configs['relations_expected_keys'],
    #         is_list=True
    #     )[0]
    #
    #     # Filtering and cleaning relationships based on existing entities
    #     entity_names = {entity["Name"] for entity in entities}
    #     filtered_relations = [relation for relation in relations if
    #                           relation['Entity1'] in entity_names and relation['Entity2'] in entity_names]
    #
    #     return {
    #         "Entities": entities,
    #         "Relationships": filtered_relations
    #     }

    # def run(self, input_text):
    #     """
    #     Parse the input text and build the initial knowledge graph.
    #     :param input_text: String, the text to be parsed.
    #     :return: Tuple (parsed data, initial graph information).
    #     """
    #     input_text = input_text.replace("\\", "\\\\")
    #     try:
    #         parsed_data = self.parse(
    #             data=input_text,
    #             system_prompt=parser_configs['system_input_parser'],
    #             expected_keys=parser_configs['input_expected_keys'],
    #         )
    #         initial_graph_info = self.build_knowledge_graph(data=parsed_data)
    #         return {"parsed_data": parsed_data, "initial_graph_info": initial_graph_info}
    #     except Exception as e:
    #         raise f"Error in run_parser: {e}"
