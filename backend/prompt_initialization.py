from chatgpt import ChatGPTHelper, ChatMessage


REASONING_COLUMN = "Reasoning"
PROMPT_INITIALIZATION_SYSTEM_PROMPT = "You are a highly skilled AI Prompt Engineer tasked with setting up a ChatGPT agent for optimal performance. Your goal is to generate a configuration that enables the ChatGPT agent to understand and execute the task efficiently."
PROMPT_INITIALIZATION_USER_PROMPT_TEMPLATE = """I want to create a ChatGPT powered agent that performs the following task: {short_task_description}
{input_feature_descriptions}

# Instructions
1. Analyze the Input:
Carefully read the provided short_task_description and input_feature_descriptions.
Identify the key objective of the task and the specific features that are important to consider.

2. Generate Task Instruction:
Write a clear and concise task instruction for the ChatGPT agent, with the template below. Use prompt engineering best practices.
Template:
# Introduction
<Assign a role that performs the task effectively to the agent and provide a brief introduction to the task the agent will perform.>
# Instructions
<Break down the task instructions to guide the agent. Be concise and do not include obvious steps>
<If it is necessary, add a ## Rules subsection that describes decision making rules for the task or areas in the input to focus on if applicable>
```json
{{
  "task_instruction": "<Your task instruction here>",
  "suggested_outputs": [
    {{
      "column_name": "<name of the output field>",
      "column_description": "<description of the output field>",
      "column_type": "<data type: STRING>"
    }}
    // Add more output columns as needed
  ],
  "suggested_examples": [
    {{
        "$#{{input_field_name1}}": "<value>",
        "$#{{input_field_name2}}": "<value>"
        "<output_column_name1>": "<expected_value>",
        "<output_column_name2>": "<expected_value>"
        "{reasoning_column_name}": "<reasoning for the outputs>"
    }}
    // Add more examples as needed
  ]
}}```"""


class PromptInitializer:
    @staticmethod
    def dump_input_feature_descriptions(input_feature_descriptions: dict) -> str:
        input_feature_context = "Input Features:\n"
        input_feature_context += "\n".join([f"- {feature_name}: {description}" for feature_name, description in input_feature_descriptions.items()])
        input_feature_context += "\nIf you think these features are not relevant to the task, please ignore them.\n"
        return input_feature_context

    @staticmethod
    def get_prompt_initialization_outputs(short_task_description: str, input_feature_descriptions: dict):
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
            response_content = chatgpt_helper.get_chat_response_content(messages)
        except:
            raise ValueError("Error occurred during LLM call for Prompt Initialization")
        json_response = chatgpt_helper.extract_json_from_response(response_content)
        if not json_response:
            raise ValueError("No JSON response found")
        
        

        
