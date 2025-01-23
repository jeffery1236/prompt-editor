from dotenv import load_dotenv
load_dotenv()
import re
import json
from openai import OpenAI
from pydantic import BaseModel
from typing import Dict, Optional


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatGPTHelper:
    def __init__(self, model: str = "gpt-4o"):
        self._client = OpenAI()
        self.model = model

    async def get_chat_response_content(self, chat_messages: list[ChatMessage]) -> str:
        chat_completion_response = self._client.chat.completions.create(
            model="gpt-4o",
            messages=[dict(message) for message in chat_messages],
            temperature = 0.2,
            top_p = 0.1,
            frequency_penalty= 0,
            presence_penalty= 0,
            max_tokens= 2000
        )

        if chat_completion_response.choices and len(chat_completion_response.choices) > 0:
            choice = chat_completion_response.choices[0]
            if choice and hasattr(choice, "message") and hasattr(choice.message, "content"):
                return choice.message.content

    def extract_json_from_response(self, content: str) -> Optional[Dict]:
        """
        Extracts a JSON object from a string content.

        Args:
            content: The string content to extract the JSON from.

        Returns:
            The extracted JSON object as a dictionary, or None if no valid JSON is found.
        """
        def get_json_from_free_text(input_text: str) -> dict:
            start = input_text.find("{")
            if start == -1:
                raise ValueError("No JSON object found in the text")

            count = 0
            in_string = False
            for i in range(start, len(input_text)):
                char = input_text[i]

                if char == '"' and input_text[i - 1] != "\\":
                    in_string = not in_string

                if not in_string:
                    if char == "{":
                        count += 1
                    elif char == "}":
                        count -= 1

                    if count == 0:
                        end = i + 1
                        break
            else:
                raise ValueError("No complete JSON object found in the text")

            json_str = input_text[start:end]
            return json.loads(json_str)

        json_pattern = re.compile(r"```json\s*(\{.*?\})\s*```|```\s*(\{.*?\})\s*```", re.DOTALL)
        match = json_pattern.search(content)
        if match:
            json_str = match.group(1) if match.group(1) else match.group(2)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                loaded_json = get_json_from_free_text(content)
                if not loaded_json:
                    raise ValueError(f"Invalid JSON format: {e}")
                return loaded_json
        return None
