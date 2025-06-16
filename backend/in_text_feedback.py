import difflib
from pydantic import BaseModel

from chatgpt import ChatGPTHelper, ChatMessage
from prompt_segmentation import PromptSegment, SECTION_TYPES
from typing import Optional, Tuple

section_wise_guidelines = {
    "Role Assignment": [
        "Ensure the role assignment is explicit and relevant to the task. Avoid ambiguous roles like 'helper' or 'assistant' without specifying the domain of expertise."
    ],
    # "Input Definitions": [
    #     "Pre-process numerical inputs into categorical or text values when the task benefits from clearer categorical reasoning.",
    #     # "Ensure that the input feature explanations and input feature values are listed in separate sections in the prompt. This helps the LLM to understand the input feature values better.",
    # ],
    "Task.Instructions": [
        "If the instructions involves complicated reasoning, break it down into manageable steps like a chain-of-thought to guide the LLM through reasoning."
    ],
    "Examples": [
        "Provide reasoning to each of the examples provided.",
        # "Ensure that examples provided represent the most common or important cases. Include edge cases to guide the LLM’s response in non-typical scenarios."
    ],
}

class TextPositions(BaseModel):
    fromLine: int  # 0-indexed
    toLine: int
    fromCh: int  # 0-indexed
    toCh: int

class InTextFeedback(TextPositions):
    feedback: str
    referenceSegment: Optional[str] = None

def dump_guidelines_json(section_wise_guidelines: dict) -> str:
    string = ""
    for section, guidelines in section_wise_guidelines.items():
        string += f"{section}:\n"
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
    
    def get_text_positions_for_multiline(
        self, multiline_str: str, ratio_threshold=0.9
    ) -> Optional[list[TextPositions]]:
        """
        Matches the multi-line string line by line in the prompt, returning a
        list of TextPositions – one per line – in consecutive order.

        If any line cannot be matched (exactly or fuzzily) in order, returns None.
        """
        lines = multiline_str.split('\n')
        positions_list: list[TextPositions] = []

        # We'll track where in self.prompt we should resume searching
        current_search_start = 0

        for line in lines:
            # If the line is blank, decide how you want to handle it.
            # Here, we skip matching logic but still return a 0-length TextPositions
            # so that each line in the segment gets an entry. 
            if not line.strip():
                # We'll just create a zero-length match at current_search_start.
                # You could also choose to skip entirely or implement a specialized match.
                pos = self._create_zero_length_text_positions(current_search_start)
                if pos is None:
                    # Could not map current_search_start => out of range
                    return None
                positions_list.append(pos)
                continue

            # 1) Exact match attempt from current_search_start onward
            found_idx = self.prompt.find(line, current_search_start)

            # 2) If exact match not found, try fuzzy match
            if found_idx == -1:
                found_idx, matched_len = self._fuzzy_match_line(line, current_search_start, ratio_threshold)
                if found_idx is None:
                    # If neither exact nor fuzzy match is found, return None
                    return None
                # If we did a fuzzy match, matched_len might differ from len(line)
                line_start = found_idx
                line_end = found_idx + matched_len
            else:
                # Exact match
                line_start = found_idx
                line_end = found_idx + len(line)

            # Create a TextPositions object for this line
            from_line, from_ch = self.get_line_and_char_id(line_start)
            # line_end - 1 => last character index for the matched line
            to_line, to_ch = self.get_line_and_char_id(line_end - 1)

            # If any mapping fails, treat it as unmatched
            if from_line is None or to_line is None:
                return None

            # Append positions for the matched line
            positions_list.append(
                TextPositions(
                    fromLine=from_line,
                    toLine=to_line,
                    fromCh=from_ch,
                    toCh=to_ch + 1  # +1 to make the end exclusive
                )
            )

            # Update current_search_start to just after the matched line
            current_search_start = line_end

        return positions_list if positions_list else None

    def _create_zero_length_text_positions(
        self, position_index: int
    ) -> Optional[TextPositions]:
        """Creates a zero-length TextPositions object at the given position_index in the prompt."""
        if not (0 <= position_index < len(self.prompt)):
            return None

        from_line, from_ch = self.get_line_and_char_id(position_index)
        if from_line is None or from_ch is None:
            return None
        # Zero-length => fromCh and toCh are the same
        return TextPositions(from_line, from_line, from_ch, from_ch)

    def _fuzzy_match_line(
        self, line: str, current_search_start: int, ratio_threshold: float
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Helper for fuzzy matching a single line in self.prompt starting at current_search_start.
        Returns (start_idx, matched_length) if found, or (None, None) otherwise.
        """

        remaining_text = self.prompt[current_search_start:]
        if len(line) > len(remaining_text):
            ratio = difflib.SequenceMatcher(None, line, remaining_text).ratio()
            if ratio >= ratio_threshold:
                return current_search_start, len(remaining_text)
            else:
                return None, None

        # Otherwise, try sliding over potential matches
        best_ratio = 0.0
        best_start = None

        for offset in range(len(remaining_text) - len(line) + 1):
            candidate = remaining_text[offset : offset + len(line)]
            ratio = difflib.SequenceMatcher(None, line, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = current_search_start + offset

        if best_start is not None and best_ratio >= ratio_threshold:
            return best_start, len(line)

        return None, None

    # def get_text_positions(self, segment_str: str, ratio_threshold=0.9) -> TextPositions:
    #     """
    #     Attempts to find 'segment_str' in 'self.prompt' with up to 'self.max_allowed_errors' edits.
    #     Returns the first match found, or raises ValueError if none is found.
    #     """
    #     # Exact match
    #     idx = self.prompt.find(segment_str)
    #     if idx != -1:
    #         start_global = idx
    #         end_global = idx + len(segment_str)

    #         from_line, from_ch = self.get_line_and_char_id(start_global)
    #         to_line, to_ch = self.get_line_and_char_id(end_global - 1)
    #         if from_line is None or to_line is None:
    #             raise ValueError(f"Failed to map exact match for '{segment_str}'.")

    #         return TextPositions(
    #             fromLine=from_line,
    #             toLine=to_line,
    #             fromCh=from_ch,
    #             toCh=to_ch + 1  # +1 to make it exclusive
    #         )

    #     # Fuzzy match
    #     segment_len = len(segment_str)
    #     best_ratio = 0.0
    #     best_start = None
        
    #     if segment_len > len(self.prompt):
    #         best_ratio = difflib.SequenceMatcher(None, segment_str, self.prompt).ratio()
    #         best_start = 0
    #     else:
    #         for start_idx in range(len(self.prompt) - segment_len + 1):
    #             candidate = self.prompt[start_idx:start_idx + segment_len]
    #             ratio = difflib.SequenceMatcher(None, segment_str, candidate).ratio()
    #             if ratio > best_ratio:
    #                 best_ratio = ratio
    #                 best_start = start_idx

    #     if best_start is not None and best_ratio >= ratio_threshold:
    #         start_global = best_start
    #         end_global = best_start + min(segment_len, len(self.prompt))

    #         from_line, from_ch = self.get_line_and_char_id(start_global)
    #         to_line, to_ch = self.get_line_and_char_id(end_global - 1)
    #         if from_line is None or to_line is None:
    #             raise ValueError(f"Failed to map approximate match for '{segment_str}'.")

    #         return TextPositions(
    #             fromLine=from_line,
    #             toLine=to_line,
    #             fromCh=from_ch,
    #             toCh=to_ch + 1
    #         )

    #     # If we reach here, no match was found within the difflib threshold
    #     raise ValueError(
    #         f"No match found for segment '{segment_str}' "
    #         f"with ratio >= {ratio_threshold:.2f}."
    #     )

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
    - If a section does not violate any guidelines, DO NOT include it in your output.

    # Output Format
    Provide your output in following JSON format:
    ```json
    [
        {{
            "type": <section type>,
            "content": <prompt segment content>,
            "feedback": <aggregated feedback>
        }}
        {{
        // add more violated segments as needed
    }}
    ]
    ```
    """
    
    chatgpt_helper = ChatGPTHelper()

    chat_messages = [
        ChatMessage(role="system", content=IN_TEXT_FEEDBACK_PROMPT.format(section_guidelines=dump_guidelines_json(section_wise_guidelines))),
        ChatMessage(role="user", content=f'Sections: {[{"type": segment.type.value, "content": segment.content} for segment in prompt_segments]}'),
    ]

    response_content = await chatgpt_helper.get_chat_response_content(chat_messages)
    json_response = chatgpt_helper.extract_json_from_response(response_content)
    if not json_response:
        print('No JSON response')
        return []

    segment_position_finder = SegmentPositionFinder(prompt)
    in_text_feedback: list[InTextFeedback] = []
    for feedback_item in json_response:
        segment_type = feedback_item["type"]
        segment_content = feedback_item["content"]
        feedback = feedback_item["feedback"]
        print(f"Type: {segment_type} { SECTION_TYPES.ROLE_ASSIGNMENT.value} Segment: {feedback_item['content']}, Feedback: {feedback_item['feedback']}")
        if not feedback:
            continue
        positions_list = segment_position_finder.get_text_positions_for_multiline(segment_content)
        if positions_list is None:
            continue

        for pos in positions_list:
            # Hacky way of inserting/'retrieving' referenceSegment for Demo
            reference_segment = None
            if segment_type == SECTION_TYPES.ROLE_ASSIGNMENT.value:
                reference_segment = "You are an expert business analyst specializing in reading invoices."
            if segment_type == SECTION_TYPES.TASK_INSTRUCTIONS.value:
                reference_segment = """
# Instructions
1. Extract key features: Identify patterns in length, format, prefixes/suffixes, sequences, and special characters from past numbers.
Analyze Patterns:
2. Determine common rules for validity, such as:
- Fixed length or range.
- Consistent prefixes/suffixes.
- Sequential or structured numeric patterns.
3. Validate Against Rules: Check the input number for conformity with identified rules. Flag deviations as invalid unless they fit rare exceptions observed in the list."""
            
            in_text_feedback.append(InTextFeedback(
                fromLine=pos.fromLine,
                toLine=pos.toLine,
                fromCh=pos.fromCh,
                toCh=pos.toCh,
                feedback=feedback,
                referenceSegment=reference_segment,
            ))

    return in_text_feedback