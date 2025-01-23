from pydantic import ValidationError, BaseModel

from chatgpt import ChatGPTHelper, ChatMessage
from prompt_segmentation import PromptSegment

class GeneralFeedback(BaseModel):
    title: str
    reasoning: str
    suggestion: str


async def get_general_feedback(prompt: str, missing_sections: list) -> list[GeneralFeedback]:
    GENERAL_FEEDBACK_PROMPT = """# Introduction
You are an expert prompt engineer responsible for analyzing a given prompt and identifying missing sections that could enhance its performance. You will receive a list of missing sections along with the prompt, and your task is to evaluate whether these missing sections are relevant. For each applicable section, you should provide feedback that includes:
   1. Section Type: The section that should be added to the prompt.
   2. Reasoning: A brief explanation of why adding this section would improve the prompt's performance. Consider how it would make the LLM's task clearer, more structured, or aligned with user expectations.
   3. Suggested Section Text: Provide a sample text for the missing section, keeping in mind the task’s context, and optimize it using the best practices outlined below along with prompt engineering best practices.

# Section Types
The following section types might be missing from the prompt, and your goal is to analyze which of these should be included. As you analyze the prompt, follow these best practices for each section:
- Role Assignment: Ensure the role assignment is explicit and relevant to the task. Avoid ambiguous roles like 'helper' or 'assistant' without specifying the domain of expertise.
- Input Definitions: Pre-process numerical inputs into categorical or text values when the task benefits from clearer categorical reasoning.
- Ensure that input feature explanations and input feature values are listed in separate sections in the prompt. This helps the LLM to understand the input feature values better.
- Task.Description: Provide a high-level task overview focusing on the core objective without including specific instructions or constraints.
- Task.Instructions: Break the task into manageable steps, especially for complex tasks. Use techniques like chain-of-thought to guide the LLM through reasoning.
- Task.Constraints: Specify word limits, forbidden terms, or other constraints to control task scope and output consistency.
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
            "title": "Missing <missing section_type that you recommend to be included>",
            "reasoning": "<concise reasoning why the section improves performance>",
            "suggestion": "<suggested section text that you recommend to be included>"
        },
        // add more sections as needed
    ]
}
```

# Examples
Input Prompt: "Summarize the key findings from the sales data for Q1 2023 in a list format."
Missing Sections: ["Role Assignment", "Task.Constraints", "Examples"]

Output:
```json
{
    "MissingSectionFeedback": [
        {
            "title": "Missing Role Assignment",
            "reasoning": "A relevant role conditions the model and ensures the response aligns with the correct perspective.",
            "suggestion": "You are a business analyst tasked with reviewing and summarizing sales data insights."
        },
        {
            "title": "Missing Examples",
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
            parsed_json_content.append(GeneralFeedback(**item))
        except ValidationError as e:
            raise ValueError(f"Failed to parse {item}: {e}")

    return parsed_json_content