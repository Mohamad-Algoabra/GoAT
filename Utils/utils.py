import ast
import os
import json
import re

from transformers import AutoTokenizer


def save_experiment(base_dir, **kwargs):
    # Create the base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Get the next experiment count
    existing_experiments = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    next_count = 1 + len(existing_experiments)

    # Create a new directory for this experiment
    experiment_dir = os.path.join(base_dir, f'experiment_{next_count}')
    os.makedirs(experiment_dir)

    # Save each item in kwargs to a separate JSON file
    for key, value in kwargs.items():
        file_name = f'{key}.json'
        with open(os.path.join(experiment_dir, file_name), 'w') as f:
            json.dump(value, f, indent=4)


def load_experiment(base_dir, experiment_number):
    # Directory for the specified experiment
    experiment_dir = os.path.join(base_dir, f'experiment_{experiment_number}')
    print(experiment_dir)
    # Check if the specified experiment directory exists
    if not os.path.exists(experiment_dir):
        raise FileNotFoundError(f"No experiment found with number: {experiment_number}")

    # Load all JSON files in the experiment directory
    experiment_data = {}
    for file in os.listdir(experiment_dir):
        if file.endswith('.json'):
            with open(os.path.join(experiment_dir, file), 'r') as f:
                key = file[:-5]  # Remove '.json' from filename
                experiment_data[key] = json.load(f)

    return experiment_data


def extract_and_validate(text, expected_keys, is_list=False):
    # Define regex patterns for dictionary and list of dictionaries
    dict_pattern = r'\{[^{}]*\}'
    list_pattern = r'\[([^]]*)\]'

    if is_list:
        # Match and process each dictionary within a potentially corrupted list
        matches = re.findall(dict_pattern, text)
        valid_dicts = []
        for match in matches:
            try:
                parsed_dict = ast.literal_eval(match)
                if isinstance(parsed_dict, dict) and set(parsed_dict.keys()) == set(expected_keys):
                    valid_dicts.append(parsed_dict)
            except (SyntaxError, ValueError):
                continue
        return valid_dicts if valid_dicts else None

    else:
        # Process a single dictionary
        match = re.search(dict_pattern, text)
        if match:
            try:
                parsed_dict = ast.literal_eval(match.group(0))
                if isinstance(parsed_dict, dict) and set(parsed_dict.keys()) == set(expected_keys):
                    return parsed_dict
            except (SyntaxError, ValueError):
                pass

    return None


def count_tokens(text, model_name="mlabonne/Beagle14-7B"):
    # Initialize the tokenizer with the specified model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize the input text and count the tokens
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    num_tokens = len(input_ids)

    return num_tokens
