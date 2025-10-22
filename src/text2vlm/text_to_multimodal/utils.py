# Copyright (c) 2025 Advai Ltd.
# Licensed under the MIT License. See LICENSE file in the project root for details.

"""
Utilities for Text to Multimodal conversion.
"""
import json
import logging
import os
from typing import Any, Optional


def load_json(filename: str) -> Optional[Any]:
    """Load JSON data from a file.

    :param filename: Path to the JSON file
    :return: JSON data if successful or None.
    """
    try:
        if os.path.exists(filename):
            with open(filename, "r") as file:
                data = json.load(file)
            return data
        raise FileNotFoundError
    except FileNotFoundError:
        logging.error(f"Error: The file {filename} was not found.")
        return None
    except json.JSONDecodeError:
        logging.error(
            f"Error: The file {filename} does not contain valid JSON."
        )
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None
