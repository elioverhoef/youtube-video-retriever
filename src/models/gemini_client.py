import os
from typing import Optional, Tuple
import google.generativeai as genai
from pathlib import Path
import yaml
import logging
import re


class GeminiClient:
    def __init__(self):
        # Load config
        config_path = Path("config/config.yaml")
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Configure logging
        logging.basicConfig(level=self.config["logging"]["level"])
        self.logger = logging.getLogger(__name__)

        # Set up Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        genai.configure(api_key=api_key)

        # Suppress GRPC warnings
        os.environ["GRPC_VERBOSITY"] = "ERROR"

    def clean_json_string(self, json_str: str) -> str:
        """Clean JSON string by removing trailing commas and formatting issues."""
        # Remove code blocks if present
        json_str = json_str.replace("```json", "").replace("```", "").strip()

        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)

        return json_str

    def get_completion(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Tuple[str, str]:
        """Get completion with automatic model fallback.
        Returns: (response_text, model_used)
        """
        # Add explicit JSON formatting instruction if prompt contains JSON
        if '"format"' in prompt or '"type"' in prompt:
            if system_prompt:
                system_prompt += "\nIMPORTANT: Ensure the JSON output is valid with NO trailing commas."
            else:
                system_prompt = "IMPORTANT: Ensure the JSON output is valid with NO trailing commas."

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        # Try primary model first
        primary = self.config["models"]["primary"]
        response = self._try_model(primary["name"], full_prompt, primary["max_retries"])
        if response and ('"format"' in prompt or '"type"' in prompt):
            response = self.clean_json_string(response)

        if response:
            return response, primary["name"]

        # Try fallback models
        for fallback in self.config["models"]["fallbacks"]:
            response = self._try_model(
                fallback["name"], full_prompt, fallback["max_retries"]
            )
            if response and ('"format"' in prompt or '"type"' in prompt):
                response = self.clean_json_string(response)

            if response:
                return response, fallback["name"]

        raise Exception("All models failed to provide a valid response")

    def _try_model(
        self, model_name: str, prompt: str, max_retries: int
    ) -> Optional[str]:
        """Try to get completion from a specific model."""
        self.logger.info(f"Trying model: {model_name}")

        # Try with all API keys
        api_keys = [
            os.getenv("GOOGLE_API_KEY"),
            os.getenv("GOOGLE_API_KEY_2"),
            os.getenv("GOOGLE_API_KEY_3"),
            os.getenv("GOOGLE_API_KEY_4"),
            os.getenv("GOOGLE_API_KEY_5"),
        ]

        for api_key in api_keys:
            if not api_key:
                continue

            response = self._try_with_api_key(model_name, prompt, api_key)
            if response:
                return response

        return None

    def _try_with_api_key(
        self, model_name: str, prompt: str, api_key: str
    ) -> Optional[str]:
        """Attempt to get completion with a specific API key."""
        max_attempts = 3  # Fixed number of attempts per API key

        genai.configure(api_key=api_key)

        for attempt in range(max_attempts):
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                if response and response.text:
                    self.logger.info(f"Success with {model_name}")
                    return response.text.strip()
            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt + 1} failed for {model_name}: {str(e)}"
                )
                if attempt == max_attempts - 1:
                    self.logger.error(
                        f"All attempts failed for {model_name} with current API key"
                    )

        return None
