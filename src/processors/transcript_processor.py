from pathlib import Path
import re
import yaml
import logging
import json
from typing import List, Dict
from ..models.gemini_client import GeminiClient


class TranscriptProcessor:
    def __init__(self):
        # Load config
        config_path = Path("config/config.yaml")
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.logger = logging.getLogger(__name__)
        self.client = GeminiClient()

        # Improved system prompt focusing on extracting meaningful insights
        # Load content model
        content_model_path = Path("config/content_model.json")
        with open(content_model_path) as f:
            self.content_model = json.load(f)

        # Generate dynamic system prompt
        self.system_prompt = self._generate_system_prompt()
        
        # Initialize user prompt
        self.user_prompt = f"""Analyze this transcript and extract insights for {self.content_model['title']}.
Focus on practical, actionable information that matches the requested attributes.

Transcript:
{{text}}"""

    def _generate_system_prompt(self) -> str:
        """Generate system prompt dynamically from content model."""
        model = self.content_model
        prompt = [
            f"You are an expert analyst extracting insights for a {model['title']}.",
            f"{model['description']}\n",
            "Focus on extracting the following attributes for each section:\n"
        ]

        # Add formatting instructions
        format_fields = model['formatting']['metadata_fields']
        prompt.append("For each insight, include:")
        for field in format_fields:
            prompt.append(f"- {field}")
        prompt.append("")

        # Add confidence scale explanation
        conf = model['formatting']['confidence_scale']
        prompt.append(f"Confidence: [{conf['min']} to {conf['max']} {conf['type']}] ({conf['description']})\n")

        # Add section-specific instructions
        prompt.append("Format your response with consistent indentation in clear markdown sections:\n")
        
        for section_name, section in model['sections'].items():
            prompt.append(f"## {section_name}")
            prompt.append(f"{section['description']}")
            
            if 'attributes' in section:
                for attr_name, attr in section['attributes'].items():
                    prompt.append(f"- **{attr_name}**: [{', '.join(attr['required_fields'])}]")
                    prompt.append(f"    {attr['description']}")
            prompt.append("")

        # Add general formatting instructions
        prompt.append("Be precise and quantitative. Include all relevant measurements and details.")
        prompt.append("Focus on extracting actionable information that matches the requested attributes.")

        return "\n".join(prompt)

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess transcript text."""
        # Remove markdown headers
        text = re.sub(r"^#.*$", "", text, flags=re.MULTILINE)

        # Clean whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.replace("\u00a0", " ")

        # Break into sentences
        text = re.sub(r"([.!?])\s+", r"\1\n", text)

        # Remove markdown formatting
        text = re.sub(r"[*_`#]", "", text)

        return text.strip()

    def process_transcript(self, transcript_path: Path) -> Dict[str, List[str]]:
        """Process a single transcript and return structured insights."""
        self.logger.info(f"Processing transcript: {transcript_path}")

        try:
            # Read transcript
            with open(transcript_path, "r", encoding="utf-8") as f:
                text = f.read()

            # Preprocess text
            text = self.preprocess_text(text)

            # Get model response
            prompt = self.user_prompt.format(text=text)
            response, model_used = self.client.get_completion(
                prompt, self.system_prompt
            )

            self.logger.info(f"Successfully processed with model: {model_used}")

            # The response is already in markdown format with sections
            return {
                "content": response,
                "source": transcript_path.stem,
                "model": model_used,
            }

        except Exception as e:
            self.logger.error(f"Error processing {transcript_path}: {str(e)}")
            return None
