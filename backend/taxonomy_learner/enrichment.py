from pydantic import BaseModel
from typing import Optional
from chatgpt import ChatGPTHelper, ChatMessage
from taxonomy_learner.models import Node

class EnrichmentOutput(BaseModel):
    should_split: bool
    reasoning: str
    child_types: list[str] = []

def get_enrichment_system_prompt() -> str:
    return """You are an LLM agent tasked with refining a taxonomy of prompt segment types.

**You will receive:**
- A batch of prompt segments, all labeled with the same segment type.
- The current schema/taxonomy of segment types (with definitions if available).

**Your goal:**
- Decide whether the current segment type should be subdivided into more specific child types.
- Provide clear justification and examples for any proposed split.

**Instructions:**
- Carefully analyze the batch of prompt segments and the definition of their type.
- If the batch contains clear subcategories, propose a split with names, definitions, and examples for each.
- If no split is needed, justify why.

**Output your response in the following JSON format:**

{
  "should_split": "<string: True if you decide to subdivide current segment type else false>",
  "reasoning": "<string: explain your chain-of-thought reasoning>",
  "child_types": [
    {
      "name": "<string: name of child type>",
      "description": "<string: clear description>",
      "examples": [
        "<string: example segment 1>",
        "<string: example segment 2>"
      ]
    }
    // Only include if decision is to split else just give an empty list
  ],
}
"""


async def enrichment(current_node: Node, example_node_segments: list[str], root: Node) -> Optional[list[Node]]:
    formatted_input_prompt = f"""Batch: {example_node_segments}\nCurrent segment type: {current_node.name}\nCurrent Schema: {root.get_formatted_taxonomy()}"""
    chat_messages = [
        ChatMessage(role="system", content=get_enrichment_system_prompt()),
        ChatMessage(role="user", content=formatted_input_prompt)
    ]

    print("enriching")
    chatgpt_helper = ChatGPTHelper()
    response_content = await chatgpt_helper.get_chat_response_content(chat_messages)
    enrichment_output = chatgpt_helper.extract_json_from_response(response_content)

    return EnrichmentOutput(**enrichment_output)
