"""
Utility functions for the Customer Churn Prediction project.
"""

import os
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    return PROJECT_ROOT


def get_data_path(filename: str = "telco_churn.csv") -> str:
    """Return the absolute path to a file in the data/ directory."""
    return os.path.join(PROJECT_ROOT, "data", filename)


def get_model_path(filename: str) -> str:
    """Return the absolute path to a file in the models/ directory."""
    path = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, filename)


def get_output_path(filename: str) -> str:
    """Return the absolute path to a file in the outputs/ directory."""
    path = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, filename)


def save_json(data: dict | list, filepath: str) -> None:
    """Save data as a formatted JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> dict | list:
    """Load data from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def print_section(title: str, char: str = "=", width: int = 60) -> None:
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")
