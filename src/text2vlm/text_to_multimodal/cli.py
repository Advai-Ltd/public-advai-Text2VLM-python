# Copyright (c) 2025 Advai Ltd.
# Licensed under the MIT License. See LICENSE file in the project root for details.


"""Provide an interactive CLI tool for handling multiple functionalities.

1. Text2VLM: Converts text datasets to VLM format.
2. Run Model: Executes one of the predefined models on given input data.
3. Classify: Classifies input data using a model.

The script provides an interactive command-line interface
where users can navigate forwards and backwards through options
and enter parameters after launch.
It includes input validation using Pydantic v2
and error handling to manage invalid or incorrect inputs,
allowing users to retry entering parameters with suggestions.
"""

import logging
import os
import sys
from typing import Annotated, Callable, Dict

from pydantic import BaseModel, Field, ValidationError

from text2vlm.text_to_multimodal import classify_refusal as cr
from text2vlm.text_to_multimodal import target_model as tm
from text2vlm.text_to_multimodal import text_to_multimodal as t2m

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)

LOCAL_SERVER_IP = "localhost"
LOCAL_SERVER_PORT = "8000"

MODEL_MAPPING: Dict[str, str] = {
    "Llava-7b": "yorickvp/llava-v1.6-mistral-7b:19be067b589d0c46689ffa7cc3ff321447a441986a7694c01225973c2eafc874",
    "Llava-34b": "yorickvp/llava-v1.6-34b:41ecfbfb261e6c1adf3ad896c9066ca98346996d7c4045c5bc944a79d430f174",
    "Vila-8b": "Llama-3-VILA1.5-8B",
    "Vila-40b": "VILA1.5-40B",
}

DATASET_MAPPING: Dict[str, str] = {
    "interpreter_attack": "interpreter",
    "mitre_attack": "mitre",
    "medsafetybench": "med",
    "toxigen": "toxi",
}

DATASET_PATHS: Dict[str, str] = {
    "interpreter_attack": "src/text2vlm/datasets/interpreter_attack.json",
    "mitre_attack": "src/text2vlm/datasets/mitre_attack.json",
    "medsafetybench": "src/text2vlm/datasets/medsafety_datasets",
    "toxigen": "src/text2vlm/datasets/toxigen_prompts",
}

RELEVANCY_MODEL = "gpt-4o-mini"


class EnvironmentValidator:
    """Handles environment validation.

    :param required_vars: A list of required environment variables.
    :raises EnvironmentError: If one or more environment variables are missing.
    """

    @staticmethod
    def check_required_env_vars(required_vars: list) -> None:
        """Check if the required environment variables are set.

        :param required_vars: A list of required environment variables.
        :raises EnvironmentError: If one or more environment variables are missing.
        """
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )


class Text2VLMInput(BaseModel):
    """Validate inputs to the Text2VLM function."""

    filepath: str
    save_name: str
    num_samples: Annotated[int, Field(ge=-1)]


class CLIHandler:
    """CLI handler with input/output abstracted."""

    def __init__(
        self, input_func: Callable = input, output_func: Callable = print
    ) -> None:
        """Initialize the CLI handler.

        :param input_func: The function used for input, default is `input`.
        :param output_func: The function used for output, default is `print`.
        """
        self.input = input_func
        self.output = output_func

    def menu(self) -> str:
        """Display the main menu and capture user input.

        :return: The user's menu selection as a string.
        """
        self.output("\nPlease choose an option:")
        self.output("1. Text2VLM")
        self.output("2. Run target model")
        self.output("3. Classify")
        return self.input(
            "Enter the number of your choice (or 'exit' to quit): "
        ).strip()

    def handle_special_commands(self, user_input: str) -> bool:
        """Handle special commands such as 'back' or 'exit'.

        :param user_input: The user's input to check for special commands.
        :return: True if a special command was executed, False otherwise.
        """
        if user_input.lower() in ["back", "menu"]:
            self.output("Returning to the main menu...")
            return True
        elif user_input.lower() == "exit":
            self.output("Exiting...")
            sys.exit()
        return False


class Text2VLMHandler:
    """Handler for Text2VLM functionality."""

    def __init__(self, datasets: dict, text2vlm_func: Callable) -> None:
        """Initialize the Text2VLM handler.

        :param datasets: A dictionary of available datasets.
        :param text2vlm_func: Callable function to process Text2VLM operation.
        """
        self.datasets = datasets
        self.text2vlm_func = text2vlm_func

    def run(self, cli: CLIHandler) -> None:
        """Run the Text2VLM handler.

        :param cli: The CLIHandler instance for user interaction.
        :raises ValueError: If user inputs an invalid dataset number.
        :raises ValidationError: If the Pydantic validation fails.
        :raises Exception: For general errors during the process.
        """
        datasets = list(self.datasets.keys()) + ["Enter a custom filepath"]
        while True:
            cli.output("\nText2VLM - Convert text datasets to VLM format.")
            cli.output("Type 'back' to return to the main menu, or 'exit' to quit.")
            for i, dataset in enumerate(datasets, start=1):
                cli.output(f"{i}. {dataset}")

            dataset_choice = cli.input(
                "Enter the number corresponding to the dataset: "
            ).strip()
            if cli.handle_special_commands(dataset_choice):
                return

            try:
                dataset_index = int(dataset_choice) - 1
                if dataset_index == len(datasets) - 1:
                    filepath = cli.input("Enter the filepath to your dataset: ").strip()
                    save_name = cli.input(
                        "Enter the shorthand for saving (e.g., 'toxi' for toxigen): "
                    ).strip()
                    if not os.path.exists(filepath):
                        cli.output(f"The file '{filepath}' does not exist.")
                        continue
                else:
                    filepath = self.datasets[datasets[dataset_index]]
                    save_name = DATASET_MAPPING[datasets[dataset_index]]

                num_samples = cli.input(
                    "Enter the number of samples to extract (-1 for all): "
                ).strip()
                inputs = Text2VLMInput(
                    filepath=filepath,
                    save_name=save_name,
                    num_samples=int(num_samples),
                )
                self.text2vlm_func(
                    inputs.filepath, inputs.save_name, inputs.num_samples
                )
                break
            except (ValueError, ValidationError) as e:
                cli.output(f"Input error: {e}")
            except Exception as e:
                logging.error(f"An error occurred: {e}")


def text2vlm(filepath: str, save_name: str, num_samples: int) -> None:
    """Handle Text2VLM conversion.

    :param filepath: Path to the dataset file.
    :param save_name: Shorthand for saving the dataset.
    :param num_samples: Number of samples to extract (-1 for all).
    :raises Exception: If an error occurs during processing.
    """

    try:
        sys_config = {
            "file_path": filepath,
            "root_directory": f"data_files/.data_{save_name}/",
            "use_cache": False,
            "number_of_samples": num_samples,
            "output_file": f"mutated_prompts/final_result_{save_name}.json",
        }
        t2m.main(sys_config)
    except Exception as e:
        logging.error(f"An error occurred during Text2VLM processing: {e}")
        raise


class ModelRunnerHandler:
    """Handler for running the target models."""

    def __init__(self, models: dict, run_model_func: Callable) -> None:
        """Initialize the ModelRunner handler.

        :param models: A dictionary of available models.
        :param run_model_func: A callable function to run the models.
        """
        self.models = models
        self.run_model_func = run_model_func

    def run(self, cli: CLIHandler) -> None:
        """Run the ModelRunner handler.

        :param cli: The CLIHandler instance for user interaction.
        :raises ValueError: If user inputs an invalid model number.
        :raises KeyError: If the model name is not found.
        :raises Exception: For general errors during the process.
        """
        models = list(self.models.keys())
        while True:
            cli.output("\nRun Target Model - Execute a model on input data.")
            cli.output("Type 'back' to return to the main menu, or 'exit' to quit.")
            for i, model in enumerate(models, start=1):
                cli.output(f"{i}. {model}")

            model_choice = cli.input(
                "Enter the number corresponding to the model: "
            ).strip()
            if cli.handle_special_commands(model_choice):
                return

            mode = cli.input("Enter the mode (text/multimodal): ").strip().lower()
            if mode not in ["text", "multimodal"]:
                cli.output("Invalid mode. Please enter 'text' or 'multimodal'.")
                continue
            if mode == "text":
                mode_str = "txt"
            else:
                mode_str = "img"

            data_path = cli.input(
                "Enter the path to the input data file (e.g., 'mutated_prompts/final_result_med.json'): "
            ).strip()
            if not os.path.isfile(data_path):
                cli.output(f"'{data_path}' is either invalid or does not exist.")
                continue
            save_path = cli.input(
                "Enter the path to the output save file, we will add the extension (e.g., 'responses/responses_med_llava7b'): "
            ).strip()
            if not save_path:
                cli.output("Invalid save path.")
                continue
            save_path = f"{save_path}_{mode_str}.json"

            try:
                model_index = int(model_choice) - 1
                model_name = models[model_index]
                input_data = cli.input(
                    "Enter the number of samples to process: "
                ).strip()
                self.run_model_func(
                    model_name, mode, data_path, save_path, int(input_data)
                )
                break
            except (ValueError, KeyError) as e:
                cli.output(f"Input error: {e}")
            except Exception as e:
                logging.error(f"An error occurred: {e}")


def run_target_model(
    model_name: str, mode: str, data_path: str, save_path: str, num_samples: int
) -> None:
    """Run a target model on the provided input data.

    :param model_name: The model to run.
    :param mode: The mode to use (txt/multimodal).
    :param data_path: The path to the input data file.
    :param num_samples: Number of samples to feed into the model.
    :raises Exception: If an error occurs during model execution.
    """
    try:
        model_id = MODEL_MAPPING[model_name]
        sys_config = {
            "data_path": data_path,
            "save_path": save_path,
            "model": model_id,
            "mode": mode,
            "n": num_samples,
            "relevancy_model": RELEVANCY_MODEL,
            "ip": LOCAL_SERVER_IP,
            "port": LOCAL_SERVER_PORT,
        }
        tm.main(sys_config)
    except Exception as e:
        logging.error(f"An error occurred while running the model: {e}")
        raise


class ClassifyHandler:
    """Handler for classification functionality."""

    def __init__(self, classify_func: Callable) -> None:
        """Initialize the Classify handler.

        :param classify_func: A callable function to classify input data.
        """
        self.classify_func = classify_func

    def run(self, cli: CLIHandler) -> None:
        """Run the Classify handler.

        :param cli: The CLIHandler instance for user interaction.
        :raises Exception: General errors during the classification process.
        """
        while True:
            cli.output("\nClassify - Classify responses using a model.")
            cli.output("Type 'back' to return to the main menu, or 'exit' to quit.")
            filepath = cli.input("Enter the filepath to the responses: ").strip()
            if cli.handle_special_commands(filepath):
                return

            try:
                if not os.path.isfile(filepath) or not filepath.endswith(".json"):
                    cli.output(f"'{filepath}' is either invalid or does not exist.")
                    continue
                self.classify_func(filepath)
                break
            except Exception as e:
                logging.error(f"An error occurred: {e}")


def classify(filepath: str) -> None:
    """Classify input data using a predefined model.

    :param data: The data to classify.
    :raises Exception: If an error occurs during classification.
    """
    try:
        cr.process_json_file(filepath)
    except Exception as e:
        logging.error(f"An error occurred during classification: {e}")
        raise


def main():
    """Entry point for the interactive CLI."""
    required_vars = ["REPLICATE_API_TOKEN", "OPENAI_API_KEY"]
    EnvironmentValidator.check_required_env_vars(required_vars)

    cli = CLIHandler()

    text2vlm_handler = Text2VLMHandler(DATASET_PATHS, text2vlm)
    model_runner_handler = ModelRunnerHandler(MODEL_MAPPING, run_target_model)
    classify_handler = ClassifyHandler(classify)

    while True:
        choice = cli.menu()

        if cli.handle_special_commands(choice):
            sys.exit()

        if choice == "1":
            text2vlm_handler.run(cli)
        elif choice == "2":
            model_runner_handler.run(cli)
        elif choice == "3":
            classify_handler.run(cli)
        else:
            cli.output("Invalid choice. Please enter a number between 1 and 3.")


if __name__ == "__main__":
    main()
