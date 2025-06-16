from pydantic import ValidationError, BaseModel

from chatgpt import ChatGPTHelper, ChatMessage
from prompt_segmentation import PromptSegment
from typing import Optional

class GeneralFeedback(BaseModel):
    title: str
    reasoning: str
    suggestion: str
    referenceSegment: Optional[str] = None

async def get_general_feedback(prompt: str, missing_sections: list[str]) -> list[GeneralFeedback]:
    GENERAL_FEEDBACK_PROMPT = """# Introduction
You are an expert prompt engineer responsible for analyzing a given prompt and identifying missing sections that could enhance its performance. You will receive a list of missing sections along with the prompt, and your task is to evaluate whether these missing sections are relevant. For each applicable section, you should provide feedback that includes:
   1. Section Type: The section that should be added to the prompt.
   2. Reasoning: A brief explanation of why adding this section would improve the prompt's performance. Consider how it would make the LLM's task clearer, more structured, or aligned with user expectations.
   3. Suggested Section Text: Provide a sample text for the missing section, keeping in mind the task’s context, and optimize it using the best practices outlined below along with prompt engineering best practices.

# Section Types
The following section types might be missing from the prompt, and your goal is to analyze which of these should be included. As you analyze the prompt, follow these best practices for each section:
- Role Assignment: Ensure the role assignment is explicit and relevant to the task. Avoid ambiguous roles like 'helper' or 'assistant' without specifying the domain of expertise.
- Input Description: If applicable, specify which types of input or which areas in the input text should we focus on. For example, for email inputs, specify to ignore irrelevant emails then provide examples of such irrelevant email types.
- Task.Description: Provide a high-level task overview focusing on the core objective without including specific instructions or constraints.
- Task.Instructions: If the instructions involves complicated reasoning, break it down into manageable steps like a chain-of-thought to guide the LLM through reasoning.
- Task.Output-format: Specify the desired output format (e.g., JSON, Markdown, table) to ensure structured and predictable results.
- Examples: Add reasoning to each of the examples provided to help guide the LLM's decision-making process. Ensure that examples represent the most common or important cases. Include edge cases to guide the LLM’s response in non-typical scenarios.

# Instructions
Given the prompt and the missing sections, follow these steps:
   1. For each missing section in the list, assess whether its inclusion is relevant to the task.
   2. If applicable, generate feedback for the missing section, explaining why it should be included and how it will enhance the prompt’s performance.
   3. Provide a suggested text for the missing section.

# Output Format
Return the result in JSON format:
```json
{
    "MissingSectionFeedback": [
        {
            "section_name": "<missing section_type that you recommend to be included>",
            "reasoning": "<concise reasoning why the section improves performance>",
            "suggestion": "<suggested section text that you recommend to be included>"
        },
        // add more sections as needed
    ]
}
```

# Examples
Input Prompt: "Summarize the key findings from the sales data for Q1 2023 in a list format."
Missing Sections: ["Role Assignment", "Examples"]

Output:
```json
{
    "MissingSectionFeedback": [
        {
            "section_name": "Role Assignment",
            "reasoning": "A relevant role conditions the model and ensures the response aligns with the correct perspective.",
            "suggestion": "You are a business analyst tasked with reviewing and summarizing sales data insights."
        },
        {
            "section_name": "Examples",
            "reasoning": "Providing examples of expected outputs, with reasoning, can guide the LLM and increase response accuracy.",
            "suggestion": "Example input: 'Sales data for Q1 shows a 10% growth.' Example output: 'Q1 sales grew by 10%, with total revenue of $1M and a retention rate of 85%.' Reasoning: This summary focuses on key performance metrics that are commonly requested."
        }
    ]
}

```"""

    chatgpt_helper = ChatGPTHelper()

    chat_messages = [
        ChatMessage(role="system", content=GENERAL_FEEDBACK_PROMPT),
        ChatMessage(role="user", content=f"Input Prompt: {prompt}\nMissing Sections: {missing_sections}\n"),
    ]

    response_content = await chatgpt_helper.get_chat_response_content(chat_messages)

    print("feedback response: " + response_content)

    json_content = chatgpt_helper.extract_json_from_response(response_content)
    if json_content is None or "MissingSectionFeedback" not in json_content:
        raise ValueError("Failed to extract JSON from response content. Response content: " + response_content)

    parsed_json_content = []
    for item in json_content["MissingSectionFeedback"]:
        try:
            mock_reference_segment = None
            if item["section_name"] == "Role Assignment":
                mock_reference_segment = "You are an expert business analyst specializing in reading invoices."
            if item["section_name"] == "Task.Output-format":
                mock_reference_segment = """# Output Format
Return the result in JSON format:
```json
{
    "invoice_id": <extracted invoice id>,
    "is_invoice_paid": <true if invoice has been paid else false>
}
```"""
            if item["section_name"] == "Task.Instructions":
                mock_reference_segment = """# Instructions
1. Extract key features: Identify patterns in length, format, prefixes/suffixes, sequences, and special characters from past numbers.
Analyze Patterns:
2. Determine common rules for validity, such as:
- Fixed length or range.
- Consistent prefixes/suffixes.
- Sequential or structured numeric patterns.
3. Validate Against Rules: Check the input number for conformity with identified rules. Flag deviations as invalid unless they fit rare exceptions observed in the list."""

            parsed_json_content.append(GeneralFeedback(
                title=f"Missing {item['section_name']}",
                reasoning=item["reasoning"],
                suggestion=item["suggestion"],
                referenceSegment=mock_reference_segment
            ))
        except KeyError as e:
            raise ValueError(f"Invalid LLM json response for missing section feedback. Item: {item}\nError: {e}")

    return parsed_json_content