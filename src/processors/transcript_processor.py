from pathlib import Path
import re
import yaml
import logging
from typing import List, Dict, Optional
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
        self.system_prompt = """You are an expert research scientist analyzing health and longevity data.
        Your task is to extract detailed insights about diet, supplements, and health interventions.
        
        Focus on extracting:
        1. Specific dietary recommendations with exact measurements
        2. Supplement protocols and dosages
        3. Health markers and their changes
        4. Scientific methodologies used
        5. Temporal relationships and durations
        6. Individual variations and responses
        
        For each insight, include:
        - Study Type: [RCT, Observational, Case Study, Expert Opinion]
        - Population: Relevant demographics or conditions
        - Timeframe: Duration needed to see effects
        - Limitations: Any caveats or constraints
        - Tags: #relevant #categories #for #filtering
        - Confidence: [Score with stars] (1⭐ to 5⭐⭐⭐⭐⭐ based on study quality, sample size, and replication)
        
        Format your response in clear markdown sections:
        
        ## Executive Summary
        [Key findings and patterns across all sections]
        
        ## Quick Reference
        [Most actionable insights in bullet points]
        
        ## Diet Insights
        - Finding: [Details with measurements]
        - Context: [Study details, population]
        - Confidence: [Score with stars] | Tags: [#tags]
        
        ## Supplements
        - Protocol: [Name, dosage, timing]
        - Effects: [Observed outcomes]
        - Context: [Study details, population]
        - Confidence: [Score with stars] | Tags: [#tags]
        
        ## Scientific Methods
        - Study Type: [Type]
        - Methodology: [Details]
        - Key Findings: [Results]
        - Limitations: [Caveats]
        
        ## Health Markers
        - Marker: [Name]
        - Change: [Quantified change]
        - Context: [Intervention details]
        - Timeframe: [Duration]
        - Confidence: [Score with stars] | Tags: [#tags]
        
        Be precise and quantitative. Include all relevant measurements, durations, and observed effects.
        Focus on extracting actionable information that could be valuable for health optimization.
        Note any synergies or conflicts between different interventions.
        
        For confidence scores, always include the stars, e.g.:
        - Confidence: 5⭐⭐⭐⭐⭐ (strong RCT evidence)
        - Confidence: 3⭐⭐⭐ (limited observational data)
        - Confidence: 1⭐ (expert opinion only)"""
        
        self.user_prompt = """Analyze this transcript and extract all relevant health and longevity insights.
        Focus on practical, actionable information that could be implemented.
        
        Transcript:
        {text}"""
        
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess transcript text."""
        # Remove markdown headers
        text = re.sub(r'^#.*$', '', text, flags=re.MULTILINE)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('\u00a0', ' ')
        
        # Break into sentences
        text = re.sub(r'([.!?])\s+', r'\1\n', text)
        
        # Remove markdown formatting
        text = re.sub(r'[*_`#]', '', text)
        
        return text.strip()
        
    def process_transcript(self, transcript_path: Path) -> Dict[str, List[str]]:
        """Process a single transcript and return structured insights."""
        self.logger.info(f"Processing transcript: {transcript_path}")
        
        try:
            # Read transcript
            with open(transcript_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Preprocess text
            text = self.preprocess_text(text)
            
            # Get model response
            prompt = self.user_prompt.format(text=text)
            response, model_used = self.client.get_completion(prompt, self.system_prompt)
            
            self.logger.info(f"Successfully processed with model: {model_used}")
            
            # The response is already in markdown format with sections
            return {
                "content": response,
                "source": transcript_path.stem,
                "model": model_used
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {transcript_path}: {str(e)}")
            return None 