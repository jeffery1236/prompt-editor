import difflib
from pydantic import BaseModel

from chatgpt import ChatGPTHelper, ChatMessage
from prompt_segmentation import PromptSegment

section_wise_guidelines = {
    "Role Assignment": [
        "Ensure the role assignment is explicit and relevant to the task. Avoid ambiguous roles like 'helper' or 'assistant' without specifying the domain of expertise."
    ],
    "Input Definitions": [
        "Pre-process numerical inputs into categorical or text values when the task benefits from clearer categorical reasoning.",
        # "Ensure that the input feature explanations and input feature values are listed in separate sections in the prompt. This helps the LLM to understand the input feature values better.",
    ],
    "Task.Instructions": [
        "Break the task into manageable steps, especially for complex tasks. Use techniques like chain-of-thought to guide the LLM through reasoning."
    ],
    "Examples": [
        "Provide reasoning to each of the examples provided.",
        "Ensure that examples provided represent the most common or important cases. Include edge cases to guide the LLM’s response in non-typical scenarios."
    ],
}

class TextPositions(BaseModel):
    fromLine: int  # 0-indexed
    toLine: int
    fromCh: int  # 0-indexed
    toCh: int

class InTextFeedback(TextPositions):
    feedback: str

def dump_guidelines_json(section_wise_guidelines: dict) -> str:
    string = ""
    for section, guidelines in section_wise_guidelines.items():
        string += f"{section}\n"
        for guideline in guidelines:
            string += f"- {guideline}\n"
    
    return string

class SegmentPositionFinder:
    def __init__(self, prompt: str, max_allowed_errors: int = 5):
        """
        Initialize the SegmentPositionFinder with a given prompt.
        Precomputes line start positions for line/char lookups.
        """
        self.prompt = prompt
        self.max_allowed_errors = max_allowed_errors

        # Precompute the global character index at the start of each line
        self.line_start_ids = []
        total_id = 0
        lines = prompt.split('\n')
        for line in lines:
            self.line_start_ids.append(total_id)
            total_id += len(line) + 1  # +1 for newline char

    def get_line_and_char_id(self, global_id: int):
        """
        Map a global prompt offset (0-based) to (line_index, char_in_line).
        Return (None, None) if out of valid range.
        """
        if not (0 <= global_id < len(self.prompt)):
            return None, None

        num_lines = len(self.line_start_ids)
        for i in range(num_lines - 1):
            start_id = self.line_start_ids[i]
            end_id = self.line_start_ids[i + 1] - 1  # exclude newline
            if start_id <= global_id <= end_id:
                return i, global_id - start_id

        # If on the final line
        last_line_start = self.line_start_ids[-1]
        if global_id >= last_line_start:
            return num_lines - 1, global_id - last_line_start

        return None, None

    def get_text_positions(self, segment_str: str, ratio_threshold=0.9) -> TextPositions:
        """
        Attempts to find 'segment_str' in 'self.prompt' with up to 'self.max_allowed_errors' edits.
        Returns the first match found, or raises ValueError if none is found.
        """
        # Exact match
        idx = self.prompt.find(segment_str)
        if idx != -1:
            start_global = idx
            end_global = idx + len(segment_str)

            from_line, from_ch = self.get_line_and_char_id(start_global)
            to_line, to_ch = self.get_line_and_char_id(end_global - 1)
            if from_line is None or to_line is None:
                raise ValueError(f"Failed to map exact match for '{segment_str}'.")

            return TextPositions(
                fromLine=from_line,
                toLine=to_line,
                fromCh=from_ch,
                toCh=to_ch + 1  # +1 to make it exclusive
            )

        # Fuzzy match
        segment_len = len(segment_str)
        best_ratio = 0.0
        best_start = None
        
        if segment_len > len(self.prompt):
            best_ratio = difflib.SequenceMatcher(None, segment_str, self.prompt).ratio()
            best_start = 0
        else:
            for start_idx in range(len(self.prompt) - segment_len + 1):
                candidate = self.prompt[start_idx:start_idx + segment_len]
                ratio = difflib.SequenceMatcher(None, segment_str, candidate).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_start = start_idx

        if best_start is not None and best_ratio >= ratio_threshold:
            start_global = best_start
            end_global = best_start + min(segment_len, len(self.prompt))

            from_line, from_ch = self.get_line_and_char_id(start_global)
            to_line, to_ch = self.get_line_and_char_id(end_global - 1)
            if from_line is None or to_line is None:
                raise ValueError(f"Failed to map approximate match for '{segment_str}'.")

            return TextPositions(
                fromLine=from_line,
                toLine=to_line,
                fromCh=from_ch,
                toCh=to_ch + 1
            )

        # If we reach here, no match was found within the difflib threshold
        raise ValueError(
            f"No match found for segment '{segment_str}' "
            f"with ratio >= {ratio_threshold:.2f}."
        )

async def get_in_text_feedback(prompt: str, prompt_segments: list[PromptSegment]) -> list[InTextFeedback]:
    IN_TEXT_FEEDBACK_PROMPT = """You are an expert prompt engineer, and your task is to analyze a given prompt broken down into sections. For each section, you will check whether it follows the provided guidelines and offer constructive feedback for any section that fails. Your feedback will be provided as a single feedback string per section, aggregating feedback from multiple guideline violations if necessary. Follow the instructions below.

    # Input
    You will receive a prompt segmented into sections in this format:
    [
        {{
            "type": "<section type e.g. Task.Instructions>",
            "content": "<section content>"
        }},
        // remaining sections
    ]

    # Section-wise guidelines
    Here are the guidelines for each section:
    {section_guidelines}

    # Instructions
    - Analyze each section individually. For each section, you will refer to the guidelines specific to the section's section types and check if any have been violated.
        - Be extremely critical when you find a guideline violation. Make sure it is applicable in the context of the section. Else, skip it.
        - Do not repeat the guideline in your feedback if it's not applicable in the context of the section.
        - Do not criticize the classification of the segment, only the content.
    - If a section violates multiple guidelines, you will aggregate the feedback into a single string.
    - If a section does not violate any guidelines, skip it.

    # Output Format
    Provide your output in following JSON format:
    ```json
    {{
        "<input prompt segment content 1>": "<aggregated feedback>",
        // skip input prompt segments 2-3 because they did not fail any guidelines
        "<input prompt segment content 4>": "<aggregated feedback>",
        // add more segments as needed
    }}
    ```
    """
    
    chatgpt_helper = ChatGPTHelper()

    chat_messages = [
        ChatMessage(role="system", content=IN_TEXT_FEEDBACK_PROMPT.format(section_guidelines=dump_guidelines_json(section_wise_guidelines))),
        ChatMessage(role="user", content=f'Input: {[{"type": str(segment.type), "content": segment.content} for segment in prompt_segments]}'),  # TODO Improve Hacky fix for enum value resolution
    ]

    response_content = await chatgpt_helper.get_chat_response_content(chat_messages)
    json_response = chatgpt_helper.extract_json_from_response(response_content)
    if not json_response:
        raise ValueError("No JSON response found")

    segment_position_finder = SegmentPositionFinder(prompt)
    in_text_feedback: list[InTextFeedback] = []
    for segment, feedback in json_response.items():
        print(f"Segment: {segment}, Feedback: {feedback}")

        for line in segment.split('\n'):
            text_positions = segment_position_finder.get_text_positions(line)

            if not text_positions:
                print("No text positions found for segment: " + line)
                continue
        
            in_text_feedback.append(InTextFeedback(
                fromLine=text_positions.fromLine,
                toLine=text_positions.toLine,
                fromCh=text_positions.fromCh,
                toCh=text_positions.toCh,
                feedback=feedback
            ))

    return in_text_feedback


