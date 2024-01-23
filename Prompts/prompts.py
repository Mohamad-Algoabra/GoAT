output_formats = {
    'input_format':
        (
            "{{\n"
            '   "Prior_Knowledge": "String contains all information from the input in clear formulation, including numbers, values, and relevant details as initial state of thinking",\n'
            '   "Question": "The main question or query identified from the input.",\n'
            '   "Domain": "The specific domain that can be used to solve the input."\n'
            "}}"
        ),
    "input_expected_keys": ["Prior_Knowledge", "Question", "Domain"],

    'thoughts_format':
        (
            "{{\n"
            '   "Thought": Short description of the thought,\n'
            '   "Action": "Text description of the action",\n'
            '   "Result": "Text description of the result",\n'
            "}}"
        ),
    "thoughts_expected_keys": ["Thought", "Action", "Result"],

    'evaluation_format':
        (
            "{{\n"
            "'Final Score': Evaluation_Score",
            "'Hint': Improvement hint"
            "}}."
        ),
    "evaluation_expected_keys": ["Final Score", "Hint"]

}
