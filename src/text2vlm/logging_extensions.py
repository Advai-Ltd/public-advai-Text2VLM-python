"""Implementation of usefull logging formatters.

Note: The json formatter, can be especially useful when you want to analyze the
logs programmatically.

# Copyright (c) 2025 Advai Ltd.
# Licensed under the MIT License. See LICENSE file in the project root for details.

"""

import json
import logging
from typing import override


class CustomTextFormatter(logging.Formatter):  # noqa: D101
    FORMATS = {
        logging.DEBUG: "%(asctime)s - %(name)s - [DEBUG] - %(pathname)s:%(lineno)d - %(message)s",  # noqa: E501
        logging.INFO: "%(asctime)s - %(name)s - [INFO] - %(message)s",
        logging.WARNING: "%(asctime)s - %(name)s - [WARNING] - %(message)s",
        logging.ERROR: "%(asctime)s - %(name)s - [ERROR] - %(message)s",
        logging.CRITICAL: "%(asctime)s - %(name)s - [CRITICAL] - %(message)s",
    }

    @override
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def json_formatter_factory() -> logging.Formatter:
    """Factor to create a json with dict config.

    see https://docs.python.org/3/library/logging.config.html
    """
    return JsonFormatter(log_level=logging.DEBUG)


class JsonFormatter(logging.Formatter):  # noqa: D101
    @override
    def __init__(self, log_level: int):
        self.log_level = log_level
        super().__init__()

    @override
    def format(self, record):
        log_record = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        if self.log_level == logging.DEBUG:
            if record.process is not None:
                log_record["process"] = str(record.process)
            if record.thread is not None:
                log_record["thread"] = str(record.thread)
            log_record.update(
                {
                    "name": record.name,
                    "pathname": record.pathname,
                    "filename": record.filename,
                    "lineno": str(record.lineno),
                }
            )

        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(log_record)
