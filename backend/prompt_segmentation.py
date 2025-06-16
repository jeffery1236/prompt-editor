from pydantic import BaseModel
from enum import Enum
from collections import defaultdict
from chatgpt import ChatGPTHelper, ChatMessage

class SECTION_TYPES(Enum):
    ROLE_ASSIGNMENT = "Role Assignment"
    INPUT_DESCRIPTION = "Input Description"
    TASK_DESCRIPTION = "Task.Description"
    TASK_INSTRUCTIONS = "Task.Instructions"
    TASK_OUTPUT_FORMAT = "Task.Output-format"
    EXAMPLES = "Examples"

class PromptSegment(BaseModel):
    type: SECTION_TYPES
    content: str

def collate_sections_from_classified_sentences(sentences_and_segment_types : list[dict]):
    sections = defaultdict(list)
    current_sections = {}

    for sentence_and_segment_type in sentences_and_segment_types:
        sentence, section_types = sentence_and_segment_type["sentence"], sentence_and_segment_type["section_types"]
        for section_type in section_types:
            if section_type in current_sections:
                current_sections[section_type] += "\n" + sentence
            else:
                current_sections[section_type] = sentence
        
        # If this sentence is the start of a new section, eject the current sections and delete them from the current_sections dict
        stored_segment_types = list(current_sections.keys())
        for stored_segment_type in stored_segment_types:
            if stored_segment_type not in section_types:
                sections[stored_segment_type].append(current_sections[stored_segment_type])
                del current_sections[stored_segment_type]

    for stored_segment_type, stored_section in current_sections.items():
        sections[stored_segment_type].append(stored_section)
    return sections

async def get_segmented_prompt(prompt: str) -> list[PromptSegment]:
    sentence_wise_segmentation_system_prompt = """# Introduction
    You are a text preprocessing and parsing expert. Your job is to take a given prompt, break it down into meaningful sentences, and classify them into one or more of the following types:
    - Role Assignment: Specifies the assumed expertise or persona of the LLM. Identify the role the LLM should adopt, such as "expert chef" or "financial analyst."
    - Input Description: Describes the input features, helping the LLM understand what inputs to expect, what defines a valid input, what to focus on etc.
    - Task.Description: Provides a high-level overview of the task, focusing on what needs to be done, but without including specific details or steps on how to execute it. Do not include concrete instructions here.
    - Task.Instructions: Outlines the instructions for how the LLM should perform the task. This section includes actions the LLM should take, but not formatting details.
    - Task.Instructions.Rules: Describes the rules or conditions that the LLM must adhere to while completing the task.
    - Task.Output-format: Describes the expected format of the output, such as JSON, Markdown, table, or list. This ensures that the LLM returns the response in the correct structure.
    - Examples: Contains few-shot examples or illustrations that show expected input-output pairs. These examples serve as guides for the LLM on how to format its response or approach the task.
    - Miscellaneous: If sentence does not fit into the section types above, place it under Miscellaneous. This could include extra context, notes, or unrelated instructions.

    Note on Python f-strings: Some prompts may contain Python f-strings where curly brackets `{}` are used to represent expressions. When you encounter curly brackets in the prompt, treat them as part of a Python expression.

    # Instructions:
    Your task is as follows:
    1. Break the input prompt into complete and meaningful sentences. Ensure each sentence is clear, distinct, and properly punctuated.
        - Non-significant sentences: Ignore non-significant words or segments e.g. ("and", "or") if they are not relevant to the agent's behavior.
    2. Analyze each sentence and classify them into one or more section types. 
        - Prioritize Context: Choose categories that best fits the overall context. For example, if a sentence mentions a feature of the input but is mostly focused on how to handle the input during the task, it belongs under "Task Instructions and Constraints."
        - If you are still unsure whether a sentence belongs to section type A or B, include both.
    3. Return the result as a JSON object in the format below:
    ```json
    [
        {
            "sentence": <sentence 1>,
            "section_types": [<category X>, // add more categories as necessary],
        },
        {
            "sentence": <sentence 2>,
            "section_types": [<category Y>, // add more categories as necessary],
        },
        // Add more sentences as necessary
    ]
    ```
    Ensure that all special characters in the keys and values of the generated JSON, such as quotation marks ("), backslashes (\), and other special symbols, are properly escaped.
    """

    """
    # Examples

    Input Prompt Text:
    You are an expert chef with over 20 years of experience, and your task is to create a recipe for a gluten-free pasta dish. The input will be a list of ingredients provided by the user, each with its corresponding quantity. You should also avoid using any ingredients that are not listed in the input. First, analyze each of the provided ingredients. Next, consider what type of cuisines can be made with the ingredients. Then, creatively brainstorm recipes. You should avoid using any gluten-containing products in the recipe. Ensure that the recipe includes no more than 5 ingredients, and that it can be prepared in under 30 minutes. Your output should be formatted in JSON with two sections: 'ingredients' and 'steps.' Each step should be numbered.

    For example:

    Input: ["pasta", "olive oil", "garlic"]
    Output: {
        "ingredients": ["pasta", "olive oil", "garlic"],
        "steps": ["Boil pasta", "Heat olive oil", "Cook garlic in oil"]
    }

    Input: {input_prompt}

    JSON Output:
    ```json
   [{'sentence': 'You are an expert chef with over 20 years of experience, and your task is to create a recipe for a gluten-free pasta dish.',
  'section_types': ['Role Assignment', 'Task.Description']},
 {'sentence': 'The input will be a list of ingredients provided by the user, each with its corresponding quantity.',
  'section_types': ['Input Description']},
 {'sentence': 'You should also avoid using any ingredients that are not listed in the input.',
  'section_types': ['Input Description']},
 {'sentence': 'First, analyze each of the provided ingredients.',
  'section_types': ['Task.Instructions']},
 {'sentence': 'Next, consider what type of cuisines can be made with the ingredients.',
  'section_types': ['Task.Instructions']},
 {'sentence': 'Then, creatively brainstorm recipes.',
  'section_types': ['Task.Instructions']},
 {'sentence': 'You should avoid using any gluten-containing products in the recipe.',
  'section_types': ['Task.Instructions']},
 {'sentence': 'Ensure that the recipe includes no more than 5 ingredients, and that it can be prepared in under 30 minutes.',
  'section_types': ['Task.Instructions']},
 {'sentence': "Your output should be formatted in JSON with two sections: 'ingredients' and 'steps.'",
  'section_types': ['Task.Output-format']},
 {'sentence': 'Each step should be numbered.',
  'section_types': ['Task.Output-format']},
 {'sentence': 'For example:', 'section_types': ['Examples']},
 {'sentence': 'Input: ["pasta", "olive oil", "garlic"]\n',
  'section_types': ['Examples']},
 {'sentence': 'Output: {\\n    \\"ingredients\\": [\\"pasta\\", \\"olive oil\\", \\"garlic\\"],\\n    \\"steps\\": [\\"Boil pasta\\", \\"Heat olive oil\\", \\"Cook garlic in oil\\"]\\n}',
  'section_types': ['Examples']}]```"""

    chat_messages = [
        ChatMessage(role="system", content=sentence_wise_segmentation_system_prompt),
        ChatMessage(role="user", content=f"Input Prompt:\n{prompt}"),
    ]
    
    chatgpt_helper = ChatGPTHelper()
    response_content = await chatgpt_helper.get_chat_response_content(chat_messages)
    print("content", response_content)
    sentences_and_segment_types = chatgpt_helper.extract_json_from_response(response_content)
    print('Sentence Segments: ', sentences_and_segment_types)
    sections = collate_sections_from_classified_sentences(sentences_and_segment_types)

    prompt_segments: list[PromptSegment] = []
    all_section_types = set([section.value for section in SECTION_TYPES])
    for section, content in sections.items():
        if section in all_section_types:
            if type(content) == str and content:
                prompt_segments.append(PromptSegment(type=section, content=content))
            elif type(content) == list:
                prompt_segments.extend([PromptSegment(type=section, content=content[i]) for i in range(len(content)) if content[i]])

    print("Prompt Segments:", prompt_segments)
    return prompt_segments


# async def get_segmented_prompt(prompt: str) -> list[PromptSegment]:
#     segmentation_system_prompt = """You are an expert in prompt segmentation. Your job is to read a given prompt text and segment it into the following categories:

# - Role Assignment: Specifies the assumed expertise or persona of the LLM. Identify the role the LLM should adopt, such as "expert chef" or "financial analyst."
# - Input Description: Provides descriptions about the meaning or features of the inputs to the task, including any statistics, ranges, or constraints related to the inputs. It should capture details like the type of input expected (e.g., a list of ingredients, a set of numbers) and specific characteristics about the inputs provided. Do not classify example input-output pairs here. They belong in the Output Descriptions and Examples section.
# - Task.Description: Provides a high-level overview of the task, focusing on what needs to be done, but without including specific details or steps on how to execute it. Do not include concrete instructions or constraints here.
# - Task.Instructions: Outlines the step-by-step instructions for performing the task. This section includes actions the LLM should take, but not constraints or formatting details.
# - Task.Constraints: Specifies any limitations, rules, or conditions that the LLM must adhere to while completing the task. This includes word limits, forbidden terms, or other task-specific restrictions.
# - Task.Output-format: Describes the expected format of the output, such as JSON, Markdown, table, or list. This ensures that the LLM returns the response in the correct structure.
# - Examples: Contains few-shot examples or illustrations that show expected input-output pairs. These examples serve as guides for the LLM on how to format its response or approach the task.
# - Miscellaneous: If sentence does not fit into the section types above, place it under Miscellaneous. This could include extra context, notes, or unrelated instructions.

# # Handling Overlapping or Ambiguous Sections:
# Prioritize Context: If a sentence could belong to multiple categories, prioritize the category that best fits the overall context of the segment. For example, if a sentence mentions a feature of the input but is mostly focused on how to handle the input during the task, it belongs under "Task Instructions and Constraints."
# Combine Sentences: Some segments may include more than one sentence. Group sentences together if they clearly belong to the same segment.
# Categorize Once: Each sentence or segment should only be categorized once. If the same content seems to fit multiple categories, apply your best judgment to place it in the most relevant one.
# Non-significant segments: Ignore non-significant words or segments e.g. ("and", "or") if they are not relevant to the agent's behavior.

# Note on Python f-strings: Some prompts may contain Python f-strings where curly brackets `{}` are used to represent expressions. When you encounter curly brackets in the prompt, treat them as part of a Python expression.

# Provide your output in following JSON format:
# ```json
# {{
#     "<prompt segment 1>": "<prompt segment 1 content text>",
#     "<prompt segment 2>": "<prompt segment 2 content text>",
#     // add more segments as needed
# }}
# ```

# # Examples:
# Input Prompt:
# You are an expert chef with over 20 years of experience and your task is to create a recipe for a gluten-free pasta dish. The input will be a list of ingredients provided by the user, each with its corresponding quantity. You should avoid using any gluten-containing products in the recipe. You should also avoid using any ingredients that are not listed in the input. Your output should be formatted in JSON with two sections: 'ingredients' and 'steps'. Each step should be numbered.

# For example:

# Input: ["pasta", "olive oil", "garlic"]
# Output: {
#     "ingredients": ["pasta", "olive oil", "garlic"],
#     "steps": ["Boil pasta", "Heat olive oil", "Cook garlic in oil"]
# }

# Segmentation Output: 
# ```json
# {"Role Assignment": "You are an expert chef with over 20 years of experience", "Descriptions of Inputs": "The input will be a list of ingredients provided by the user, each with its corresponding quantity.", "Task.Description": "your task is to create a recipe for a gluten-free pasta dish.", "Task.Constraints": "You should avoid using any gluten-containing products in the recipe.", "Task.Output-format": "Your output should be formatted in JSON with two sections: 'ingredients' and 'steps'. Each step should be numbered.", "Examples": "For example:\n\nInput: [\"pasta\", \"olive oil\", \"garlic\"]\nOutput: {\\n    \\"ingredients\\": [\\"pasta\\", \\"olive oil\\", \\"garlic\\"],\\n    \\"steps\\": [\\"Boil pasta\\", \\"Heat olive oil\\", \\"Cook garlic in oil\\"]\\n}", "Miscellaneous": ""}
# ```
# """

#     chat_messages = [
#         ChatMessage(role="system", content=segmentation_system_prompt),
#         ChatMessage(role="user", content=f"Input Prompt:\n{prompt}"),
#     ]
    
#     chatgpt_helper = ChatGPTHelper()
#     response_content = await chatgpt_helper.get_chat_response_content(chat_messages)
#     json_response = chatgpt_helper.extract_json_from_response(response_content)
#     print("JSON Response1:", json_response)
#     prompt_segments = []
#     all_section_types = set([section.value for section in SECTION_TYPES])
#     for section, content in json_response.items():
#         if section in all_section_types:
#             if type(content) == str and content:
#                 prompt_segments.append(PromptSegment(type=section, content=content))
#             elif type(content) == list:
#                 prompt_segments.extend([PromptSegment(type=section, content=content[i]) for i in range(len(content)) if content[i]])

#     print("Prompt Segments:", prompt_segments)
#     return prompt_segments
