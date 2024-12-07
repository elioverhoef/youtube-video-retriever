import os
from typing import Optional, Tuple
import google.generativeai as genai
from pathlib import Path
import yaml
import logging

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
        
    def get_completion(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, str]:
        """Get completion with automatic model fallback.
        Returns: (response_text, model_used)
        """
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        # Try primary model first
        primary = self.config["models"]["primary"]
        response = self._try_model(primary["name"], full_prompt, primary["max_retries"])
        if response:
            return response, primary["name"]
            
        # Try fallback models
        for fallback in self.config["models"]["fallbacks"]:
            response = self._try_model(fallback["name"], full_prompt, fallback["max_retries"])
            if response:
                return response, fallback["name"]
                
        raise Exception("All models failed to provide a valid response")
        
    def _try_model(self, model_name: str, prompt: str, max_retries: int) -> Optional[str]:
        """Try to get completion from a specific model with retries."""
        self.logger.info(f"Trying model: {model_name}")
        
        for attempt in range(max_retries):
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                if response and response.text:
                    self.logger.info(f"Success with {model_name}")
                    return response.text.strip()
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {model_name}: {str(e)}")
                if attempt == max_retries - 1:
                    self.logger.error(f"All attempts failed for {model_name}")
                    return None
                    
        return None 