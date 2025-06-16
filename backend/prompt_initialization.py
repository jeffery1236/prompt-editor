from chatgpt import ChatGPTHelper, ChatMessage


REASONING_COLUMN = "Reasoning"
PROMPT_INITIALIZATION_SYSTEM_PROMPT = "You are a highly skilled AI Prompt Engineer tasked with setting up a ChatGPT agent for optimal performance. Your goal is to generate a configuration that enables the ChatGPT agent to understand and execute the task efficiently."
PROMPT_INITIALIZATION_USER_PROMPT_TEMPLATE = """I want to configure a prompt for a Large Language Model (LLM) that performs the following task: {short_task_description}
{input_feature_descriptions}

# Instructions
1. Analyze the Input:
Carefully read the provided short_task_description and input_feature_descriptions.
Identify the key objective of the task and the specific features that are important to consider.

2. Generate Task Instruction:
Write a clear and concise task instruction for the LLM, with the template below. Use prompt engineering best practices.
Template:
# Introduction
<Assign a role that performs the task effectively to the LLM and provide a brief introduction to the task the agent will perform.>

# Input Description
<Describes the input features, helping the LLM understand what inputs to expect, what defines a valid input etc.>

# Instructions
<If the instructions involves complicated reasoning, break it down into manageable steps like a chain-of-thought to guide the LLM through reasoning. Be concise and do not include obvious steps>
<If it is necessary, add a ## Rules subsection that describes decision making rules for the task or areas in the input to focus on if applicable>

## Output Format
<Instruct the LLM to generate its outputs in json format with a reasoning field e.g. {{
    "<name of output_field 1>": "<description of the output field 1>",
    "<name of output_field 2>": "<description of the output field 2>",
    "{reasoning_column_name}": "Reasoning for the outputs,
}}>

3. Propose Suggested Outputs:
Identify and suggest output columns that the ChatGPT agent should generate. Do not include unnecessary fields such as:
- Fields that provide explanation or reasoning for other output columns
- Identifier fields that are not directly used as the agent's predictions

4. Create Suggested Examples:
Develop a set of few-shot examples that demonstrate how the task should be performed. Brainstorm different scenarios and edge cases that could occur for this task and make sure your examples cover them.
Limit the number of examples to 3.
The examples should be structured as a list of dictionaries.
Each dictionary should represent an example and include keys for input_fields (with their values), their expected_output_fields and the reasoning for the outputs.
Make sure the output columns in the examples are aligned with your proposed outputs.

Generate the output in the following JSON format:
```json
{{
  "task_instruction": "<Your task instruction here>",
  "suggested_examples": [
    {{
        "$#{{input_field_name1}}": "<value>",
        "$#{{input_field_name2}}": "<value>"
        ...
        "<output_column_name1>": "<expected_value>",
        "<output_column_name2>": "<expected_value>"
        ...
        "{reasoning_column_name}": "<reasoning for the outputs>"
    }}
    // Add more examples as needed
  ]
}}```"""


class PromptInitializer:
    @staticmethod
    def _dump_input_feature_descriptions(input_feature_descriptions: dict) -> str:
        input_feature_context = "Input Features:\n"
        input_feature_context += "\n".join([f"- {feature_name}: {description}" for feature_name, description in input_feature_descriptions.items()])
        input_feature_context += "\nIf you think these features are not relevant to the task, please ignore them.\n"
        return input_feature_context
    
    @staticmethod
    def dump_examples_to_string(examples: list[dict]) -> str:
        example_str = "# Examples:"
        for i, example in enumerate(examples):
            example_str += f"\nExample {i+1}:\n"
            example_str += "\n".join([f"- {key}: {value}" for key, value in example.items()])
        return example_str

    @staticmethod
    async def get_prompt_initialization_outputs(short_task_description: str, input_feature_descriptions: dict) -> str:
        user_prompt = PROMPT_INITIALIZATION_USER_PROMPT_TEMPLATE.format(
            short_task_description=short_task_description,
            input_feature_descriptions=input_feature_descriptions,
            reasoning_column_name=REASONING_COLUMN,
        )

        chatgpt_helper = ChatGPTHelper()
        messages = [
            ChatMessage(role="system", content=PROMPT_INITIALIZATION_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt)
        ]

        try:
            response_content = await chatgpt_helper.get_chat_response_content(messages)
        except:
            raise ValueError("Error occurred during LLM call for Prompt Initialization")
        json_response = chatgpt_helper.extract_json_from_response(response_content)
        if not json_response:
            raise ValueError("No JSON response found")
        
        example_str = PromptInitializer.dump_examples_to_string(json_response["suggested_examples"])

        return json_response["task_instruction"] + "\n\n" + example_str
        


        
