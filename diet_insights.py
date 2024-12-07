from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Iterator, Union
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import json
from collections import defaultdict
import re
import asyncio
from tqdm import tqdm
import time
from itertools import islice
import os
from dotenv import load_dotenv
from ai_service import AIService

# Load environment variables
load_dotenv()

@dataclass
class DietaryEvidence:
    """Evidence supporting a dietary insight with specific measurements and context."""
    measurement: Optional[str]  # e.g. "2.6g per day"
    context: str  # The surrounding context
    impact: Optional[str]  # Observed impact/correlation
    source_video: str
    
    def to_dict(self):
        """Convert the evidence to a dictionary for JSON serialization."""
        return {
            'measurement': self.measurement,
            'context': self.context,
            'impact': self.impact,
            'source_video': self.source_video
        }

@dataclass
class DietaryInsight:
    """A meaningful dietary insight with supporting evidence and relationships."""
    food_or_nutrient: str
    key_finding: str
    evidence: List[DietaryEvidence]
    biomarkers: List[str]  # Affected biomarkers
    confidence: float
    intervention_type: str  # e.g. "supplementation", "diet_modification", "experiment"
    
    def to_dict(self):
        """Convert the insight to a dictionary for JSON serialization."""
        return {
            'food_or_nutrient': self.food_or_nutrient,
            'key_finding': self.key_finding,
            'evidence': [e.to_dict() for e in self.evidence],
            'biomarkers': self.biomarkers,
            'confidence': self.confidence,
            'intervention_type': self.intervention_type
        }

class InsightExtractor:
    """Extracts research-grade dietary insights using Open Router API."""
    
    def __init__(self, chunk_size=15000):
        self.ai_service = AIService()
        self.chunk_size = chunk_size
        
        self.extraction_prompt = """You are a research scientist analyzing dietary and longevity data.
        Analyze this text and extract any dietary or nutritional insights. Focus on:
        1. Foods, supplements, or nutrients mentioned with specific measurements
        2. Their effects on health, longevity, or biomarkers
        3. Study conditions or context
        4. Statistical findings
        5. Limitations or caveats

        For each insight found, provide:
        - Food/Nutrient: The specific food, supplement, or nutrient
        - Measurement: The dosage, level, or quantity (if any)
        - Impact: The observed effects or correlations
        - Context: The study conditions or important context
        - Confidence: How strong the evidence is (low/medium/high)
        
        Only extract insights that have clear measurements or quantitative data.
        Format each insight as JSON. Return [] if no clear insights found.

        Text: {text}
        
        Insights:"""

    def preprocess_text(self, text: str) -> str:
        """Preprocess text to make it more suitable for analysis."""
        # Remove markdown headers
        text = re.sub(r'^#.*$', '', text, flags=re.MULTILINE)
        
        # Remove empty lines
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # Clean up whitespace
        text = text.replace('\u00a0', ' ')
        text = re.sub(r'\s+', ' ', text)
        
        # Try to break into sentences more effectively
        text = re.sub(r'([.!?])\s+', r'\1\n', text)
        
        # Remove any remaining markdown formatting
        text = re.sub(r'[*_`#]', '', text)
        
        return text.strip()

    def extract_biomarkers(self, text: str) -> List[str]:
        """Extract mentioned biomarkers and their changes."""
        biomarker_patterns = [
            r'(glucose|hdl|ldl|triglycerides|creatinine|bun|uric acid|albumin|alt|ast)',
            r'(hscrp|crp|homocysteine|blood pressure|nad|cholesterol)',
            r'(horvath|phenoage|dnampacage|epigenetic age)'
        ]
        
        biomarkers = []
        for pattern in biomarker_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            biomarkers.extend(match.group(1).lower() for match in matches)
        return list(set(biomarkers))

    async def generate_insight(self, text: str) -> List[Dict]:
        """Generate research-focused insights using AI service."""
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text:
            return []
            
        prompt = self.extraction_prompt.format(text=text)
        
        response_text = ""
        try:
            async for chunk in self.ai_service.get_completion(prompt):
                response_text += chunk
                
            if response_text:
                try:
                    # Clean up the response text by removing markdown code blocks
                    insights_text = response_text.strip()
                    insights_text = re.sub(r'^```json\s*|\s*```$', '', insights_text, flags=re.MULTILINE)
                    insights_text = insights_text.strip()
                    
                    if not insights_text:
                        return []
                        
                    # Handle both array and single object responses
                    insights = json.loads(insights_text)
                    if isinstance(insights, dict):
                        insights = [insights]
                    return insights
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse LLM response as JSON: {insights_text[:100]}...")
                    return []
        except Exception as e:
            print(f"Error generating insight: {str(e)}")
            return []

    async def process_transcript(self, text: str, video_title: str) -> List[DietaryInsight]:
        """Process a single transcript to extract dietary insights."""
        if not text or not video_title:
            print(f"Warning: Empty text or video title for {video_title}")
            return []
            
        text = self.preprocess_text(text)
        
        # If text is within chunk size, process it all at once
        if len(text) <= self.chunk_size:
            insights = await self.generate_insight(text)
        else:
            # Split into chunks and process separately
            chunks = [text[i:i + self.chunk_size] 
                     for i in range(0, len(text), self.chunk_size)]
            all_insights = []
            for chunk in chunks:
                chunk_insights = await self.generate_insight(chunk)
                all_insights.extend(chunk_insights)
            insights = all_insights
            
        # Convert raw insights to DietaryInsight objects
        dietary_insights = []
        for insight in insights:
            try:
                evidence = [DietaryEvidence(
                    measurement=insight.get('measurement'),
                    context=insight.get('context', ''),
                    impact=insight.get('impact'),
                    source_video=video_title
                )]
                
                dietary_insights.append(DietaryInsight(
                    food_or_nutrient=insight.get('food_or_nutrient', ''),
                    key_finding=insight.get('key_finding', ''),
                    evidence=evidence,
                    biomarkers=self.extract_biomarkers(insight.get('context', '')),
                    confidence=0.8 if insight.get('confidence', '').lower() == 'high' else
                             0.5 if insight.get('confidence', '').lower() == 'medium' else 0.2,
                    intervention_type=insight.get('intervention_type', 'unknown')
                ))
            except Exception as e:
                print(f"Error converting insight: {str(e)}")
                continue
                
        return dietary_insights