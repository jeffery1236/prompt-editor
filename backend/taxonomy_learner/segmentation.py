from collections import defaultdict
from taxonomy_learner.models import Node
from chatgpt import ChatGPTHelper, ChatMessage
from taxonomy_learner.models import Node


def get_segmentation_system_prompt(child_nodes: list[Node]) -> str:
    formated_categories = []
    for node in child_nodes:
        formated_categories += node.get_formatted_leaf_nodes()
    categories_str = "\n- ".join(formated_categories)
    return f"""You are a text preprocessing and parsing expert. Your job is to take a given prompt, break it down into meaningful sentences, and classify them into one or more of the following categories:
    - {categories_str}
    - Miscellaneous: If sentence does not fit into the section types above, place it under Miscellaneous. This could include extra context, notes, or unrelated instructions.

    Note on Python f-strings: Some prompts may contain Python f-strings where curly brackets `{{}}` are used to represent expressions. When you encounter curly brackets in the prompt, treat them as part of a Python expression.

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
        {{
            "sentence": <sentence 1>,
            "categories": [<category X>, // add more categories as necessary],
        }},
        {{
            "sentence": <sentence 2>,
            "categories": [<category Y>, // add more categories as necessary],
        }},
        // Add more sentences as necessary
    ]
    ```
    Ensure that all special characters in the keys and values of the generated JSON, such as quotation marks ("), backslashes (\), and other special symbols, are properly escaped.
    """

def collate_sections_from_classified_sentences(sentences_and_segment_types : list[dict]):
    sections = defaultdict(list)
    current_sections = {}

    for sentence_and_segment_type in sentences_and_segment_types:
        sentence, section_types = sentence_and_segment_type["sentence"], sentence_and_segment_type["categories"]
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

async def segmentation(prompt_segment: str, child_nodes: list[Node]) -> dict[str, str]:
    chat_messages = [
        ChatMessage(role="system", content=get_segmentation_system_prompt(child_nodes)),
        ChatMessage(role="user", content=f"Input prompt segment:\n{prompt_segment}\nOutput JSON:\n"),
    ]

    print("segmenting...")
    chatgpt_helper = ChatGPTHelper()
    response_content = await chatgpt_helper.get_chat_response_content(chat_messages)
    sentences_and_segment_types = chatgpt_helper.extract_json_from_response(response_content)
    if sentences_and_segment_types:
        sections = collate_sections_from_classified_sentences(sentences_and_segment_types)
        print('Sections: ', sections)
        return sections
    else:
        return None

import os
import asyncio
from tqdm import tqdm
import json
def get_prompt_corpus(directory_path: str = "../dataset/prompts_code_understanding") -> list[str]:
    def read_prompt(path : str):
        with open(path, "r") as f:
            return f.read()

    paths = [os.path.join(directory_path, item) for item in os.listdir(directory_path)]

    return [read_prompt(path) for path in paths]

async def main() -> dict[str, str]:
    corpus = list(set(get_prompt_corpus()))
    # return await segmentation(corpus[0], ["Role Assignment", "Input Description", "Task", "Examples"])
    nodes = [
        Node(name="Role Assignment", description="""Specifies the assumed expertise or persona of the LLM. Identify the role the LLM should adopt, such as "expert chef" or "financial analyst."""),
        Node(name="Input Description", description="""Describes the input features, helping the LLM understand what inputs to expect, what defines a valid input, what to focus on etc."""),
        Node(name="Task.Description", description="""Provides a high-level overview of the task, focusing on what needs to be done, but without including specific details or steps on how to execute it. Do not include concrete instructions here."""),
        Node(name="Task.Instructions", description="""Outlines the instructions for how the LLM should perform the task. This section includes actions the LLM should take, but not formatting details."""),
        Node(name="Task.Instructions.Rules", description="""Describes the rules or conditions that the LLM must adhere to while completing the task."""),
        Node(name="Task.Instructions.Output-format", description="""Describes the expected format of the output, such as JSON, Markdown, table, or list. This ensures that the LLM returns the response in the correct structure."""),
        Node(name="Examples", description="""Contains few-shot examples or illustrations that show expected input-output pairs. These examples serve as guides for the LLM on how to format its response or approach the task."""),
    ]

    segmented_prompts = []
    count = 0
    for prompt in tqdm(corpus):
        segments = await segmentation(prompt, nodes)
        if segments:
            segmented_prompts.append(segments)
        else:
            print(f"Prompt {count} failed")
        
        count += 1

    return segmented_prompts


if __name__ == "__main__":    
    res = asyncio.run(main())
    print("done0")

    with open("segmented_prompts.json", "w") as f:
        json.dump(res, f, indent=3)
