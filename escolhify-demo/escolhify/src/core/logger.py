# -*- coding: utf-8 -*-
"""Define base logger used in the library."""

import logging
import os

from rich.logging import RichHandler


class CustomFormatter(logging.Formatter):
    """Custom formatter for the logger."""

    def format(self, record):
        """Format the log record.

        Args:
            record: The log record to format.
        """
        log_format = "[%(asctime)s] [%(levelname)s] [%(module)s] - %(message)s"
        formatter = logging.Formatter(log_format)
        return formatter.format(record)  # noqa: FS002


class FileHandlerWithRotation(logging.FileHandler):
    """File handler with log rotation."""

    def __init__(self, filename: str):
        """Initialize the file handler.

        Args:
            filename: The log filename
        """
        if not os.path.exists("logs"):
            os.makedirs("logs")
        super().__init__(os.path.join("logs", filename), encoding="utf-8")
        self.setFormatter(CustomFormatter())


class EscolhifyLogger:
    """Define the logger for the library."""

    def __init__(self, name="escolhify-logger", level=logging.DEBUG, filename="escolhify.log"):
        """Initialize the logger.

        Args:
            name: The logger name.
            level: The logging level.
            filename: The log filename.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Avoid duplicate handlers
        if not self.logger.hasHandlers():
            console_handler = RichHandler(rich_tracebacks=True, markup=True)
            file_handler = FileHandlerWithRotation(filename)

            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

    def get_logger(self):
        """Return the logger."""
        return self.logger
