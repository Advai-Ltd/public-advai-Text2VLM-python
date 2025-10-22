# Copyright (c) 2025 Advai Ltd.
# Licensed under the MIT License. See LICENSE file in the project root for details.


import json
import logging
import os
import re
from typing import Dict, List, Optional

import litellm
from litellm import completion
from litellm.cost_calculator import completion_cost

from text2vlm.text_to_multimodal.utils import load_json

litellm.set_verbose = False
logging.basicConfig(level=logging.ERROR)


class ClassifyCyberResponseService:
    def __init__(
        self,
        mode: str,
        system_message: str,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ) -> None:
        self.mode = mode
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.system_message = self.load_system_message(system_message)
        self._running_cost = 0
        self.model = model

        if not self.api_key:
            raise ValueError(
                "API key must be provided or set in the env as OPENAI_API_KEY."
            )

    def load_system_message(self, file_path) -> str:
        """Load the system message from a text file."""
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    @property
    def get_running_cost(self) -> float:
        """Get the total cost of completions."""
        return self._running_cost

    def get_completion_cost(self, response) -> float:
        """Calculate and accumulate completion cost."""
        cost = completion_cost(completion_response=response)
        self._running_cost += cost
        return cost

    def perform_inference(self, text_to_classify: str) -> str:
        """Perform inference using the specified model and returns the result.

        :param text_to_classify: The text to be classified.
        :return: The classification result string.
        """
        response = completion(
            model=self.model,
            temperature=0,
            num_retries=3,
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": text_to_classify},
            ],
            stream=False,
        )

        self.get_completion_cost(response)

        if response is None:
            raise ValueError("No response from the classifier.")

        return response["choices"][0]["message"]["content"]

    def classify_text(self, text_to_classify: str, summary: str) -> Dict[str, str]:
        """Classifies the given text using gpt-4o-mini and returns the classification result.

        :param text_to_classify: The text to be classified.
        :return: The classification result string.
        :raises ValueError: If the input text is not a string or is empty.
        """
        # Validate input
        if not isinstance(text_to_classify, str):
            raise ValueError("Input must be a string.")
        if not text_to_classify.strip():
            return {"Error": "No text to classify."}

        if self.mode == "relevance":
            prompt = f"""<Question>\n{summary}\n</Question>\n\n<Response>\n{text_to_classify}\n</Response>"""
        elif self.mode == "safety":
            prompt = text_to_classify
        else:
            raise ValueError("Invalid mode specified.")

        response: str = self.perform_inference(prompt)

        classifier_response = response
        return self.parse_classification_response(classifier_response)

    def parse_classification_response(self, response_str: str) -> Dict[str, str]:
        """Parse the classification JSON response from OpenAI and extracts the classification string.

        :param response_str: The string containing JSON-formatted classification response.
        :return: The classification string.
        :raises ValueError: If the response string is not a valid JSON or does not contain the expected structure.
        """
        try:
            # Remove ```json``` tags if they are present
            response_str = re.sub(r"```json\s*|```", "", response_str).strip()

            # Ensure the cleaned string is not empty before attempting to parse
            if not response_str:
                raise ValueError("The cleaned JSON string is empty.")

            # Convert the JSON-formatted string to a Python dictionary
            response_json = json.loads(response_str)
            return response_json
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Error decoding JSON: {e}")
            return {"Error": "Could not classify"}

    def run(self, text_to_classify: str, summary: str) -> Dict[str, str]:
        """Run the classification service.

        :param text_to_classify: The text that needs to be classified.
        :return: The classification result as a string.
        :raises ValueError: If the input text is not valid.
        """
        return self.classify_text(text_to_classify, summary)


def load_responses_from_json(file_path: str) -> Optional[List[Dict[str, str]]]:
    """Load responses from JSON file and return them as a list of dictionaries.

    :param file_path: Path to the JSON file.
    :return: List of dictionaries if successful, or None if an error occurs.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            # Load the JSON data from the file
            data = json.load(file)

            # Check that the data is a list of dictionaries
            if not isinstance(data, list):
                raise ValueError("JSON data is not a list.")

            # Ensure each item has a 'response' key
            for item in data:
                if "response" not in item:
                    raise ValueError(
                        "Items in the JSON are missing the 'response' field."
                    )

            return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from file '{file_path}'. {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return None


def save_responses_to_json(file_path: str, data: List[Dict[str, str]]) -> None:
    """Save the updated responses to a JSON file.

    :param file_path: Path to the JSON file.
    :param data: The data to save.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"An error occurred while saving the JSON file: {e}")


def load_summary(file_path: str) -> list[str]:
    json_data = load_json(file_path)
    if json_data is None:
        raise ValueError("Error loading summary.")
    summary = [(item["summary"]) for item in json_data]
    return summary


def main(config) -> None:
    responses_list: List[Dict[str, str]] | None = load_responses_from_json(
        config["file_path"]
    )

    service = ClassifyCyberResponseService(
        mode=config["mode"],
        system_message=config["system_message"],
    )

    summary_list = load_summary(config["summary_file"])

    if responses_list is not None:
        print("Responses loaded successfully:")
        count = 0
        for response_item in responses_list:
            summary = summary_list[count]
            count += 1
            print(f"Classifiying {count}/{len(responses_list)}")
            response_text: str = response_item["response"]
            classification_result = service.run(response_text, summary)

            # Add the classification result to the response item
            response_item.update(classification_result)

        # Save the updated responses back to the original JSON file
        save_responses_to_json(config["file_path"], responses_list)
    else:
        print("Failed to load responses.")

    print(f"Total cost of completions: ${service.get_running_cost}")


if __name__ == "__main__":
    # Example configuration - update with your file paths
    config: Dict[str, str] = {
        "file_path": "responses_vicuna_vila40b_img.json",
        "mode": "safety",
        "system_message": "src/text2vlm/text_to_multimodal/prompts/classify_safety.txt",
        "summary_file": "final_result_vicunabench.json",
    }

    main(config)
