import json
from typing import Any


def to_pretty_json(data: Any) -> str:
    """Return a nicely formatted JSON string."""
    return json.dumps(data, indent=2, default=str)


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dict into dot-separated keys."""
    items: list = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def sanitize_output(data: dict) -> dict:
    """Remove None values and strip whitespace from string values."""
    cleaned = {}
    for k, v in data.items():
        if v is None:
            continue
        if isinstance(v, str):
            v = v.strip()
        if isinstance(v, dict):
            v = sanitize_output(v)
        cleaned[k] = v
    return cleaned
