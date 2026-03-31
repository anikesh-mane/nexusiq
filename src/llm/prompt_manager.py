"""
Prompt Manager — loads YAML prompt templates and formats them.
"""
from pathlib import Path
from typing import Any
import yaml
from loguru import logger
from src.config import config



class PromptManager:
    """Load and render prompt templates from the /prompts directory."""

    def __init__(self, prompts_dir: Path | None = None):
        self._dir = prompts_dir or config.PROMPTS_PATH
        self._cache: dict[str, str] = {}

    def _load(self, template_name: str) -> str:
        """Load a YAML template file and cache it."""
        if template_name in self._cache:
            return self._cache[template_name]

        file_path = self._dir / f"{template_name}.yaml"
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {file_path}")

        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Expect a single key ending in "_template"
        key = next((k for k in data if k.endswith("_template")), None)
        if key is None:
            raise ValueError(f"No '*_template' key found in {file_path}")

        self._cache[template_name] = data[key]
        logger.debug(f"Loaded prompt template: {template_name}")
        return self._cache[template_name]

    def render(self, template_name: str, **kwargs: Any) -> str:
        """Render a template with the given variables."""
        template = self._load(template_name)
        return template.format(**kwargs)


# Singleton
prompt_manager = PromptManager()
