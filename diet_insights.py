from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import json
from collections import defaultdict
import re
import requests
from tqdm import tqdm

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

class OllamaInsightExtractor:
    """Extracts research-grade dietary insights using Ollama."""
    
    def __init__(self, model="mistral"):
        self.model = model
        self.base_url = "http://localhost:11434/api/generate"
        
        # Carefully crafted prompts for research-quality extraction
        self.extraction_prompt = """You are a medical researcher analyzing dietary data.
        Extract key insights about {food_or_nutrient} from this text, focusing on:
        1. Specific measurements and dosages
        2. Observed effects on biomarkers
        3. Experimental conditions
        4. Statistical findings (p-values, correlations)
        5. Any caveats or limitations
        
        Text: {text}
        
        Insight:"""

    def generate_insight(self, text: str, food_or_nutrient: str) -> Optional[str]:
        """Generate a research-focused insight using Ollama."""
        # Preprocess text to remove potential formatting issues
        text = re.sub(r'\s+', ' ', text).strip()
        food_or_nutrient = re.sub(r'\s+', ' ', food_or_nutrient).strip()
        
        if not text or not food_or_nutrient:
            return None
            
        prompt = self.extraction_prompt.format(
            food_or_nutrient=food_or_nutrient,
            text=text
        )
        
        try:
            response = requests.post(
                self.base_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1  # Low temperature for more focused analysis
                    }
                }
            )
            
            if response.status_code == 200:
                return response.json().get('response', '').strip()
            return None
        except Exception as e:
            print(f"Error generating insight: {str(e)}")
            return None

    def extract_numeric_data(self, text: str) -> List[Dict[str, str]]:
        """Extract numeric measurements and their context."""
        patterns = [
            r'(\d+(?:\.\d+)?)\s*(g|mg|mcg|IU|%|grams?|milligrams?|micrograms?)\s*(?:per|a|each)?\s*day',
            r'(\d+(?:\.\d+)?)\s*(g|mg|mcg|IU|%|grams?|milligrams?|micrograms?)\s*daily',
            r'(\d+(?:\.\d+)?)\s*(g|mg|mcg|IU|%|grams?|milligrams?|micrograms?)\s*intake'
        ]
        
        findings = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Get surrounding context (50 chars before and after)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                findings.append({
                    'measurement': match.group(0),
                    'context': text[start:end].strip()
                })
        return findings

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

    def extract_statistical_findings(self, text: str) -> List[Dict[str, str]]:
        """Extract statistical relationships and correlations."""
        patterns = [
            r'([pP]-value\s*(?:<|>|=|≤|≥)\s*0*\.?\d+)',
            r'correlation(?:\s+coefficient)?\s*(?:of|=|:)\s*(-?\d+\.?\d*)',
            r'significantly\s+(?:correlated|associated)\s+with\s+([^\.]+)'
        ]
        
        findings = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Get surrounding context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                findings.append({
                    'statistic': match.group(0),
                    'context': text[start:end].strip()
                })
        return findings

    def process_transcript(self, text: str, video_title: str) -> List[DietaryInsight]:
        """Process a single transcript to extract dietary insights."""
        if not text or not video_title:
            return []
            
        # Clean input text
        text = text.replace('\u00a0', ' ')  # Replace non-breaking spaces
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        insights = []
        
        # First pass: Extract all numeric measurements related to foods/supplements
        measurements = self.extract_numeric_data(text)
        
        # For each measurement, analyze the surrounding context
        for measurement in measurements:
            # Generate insight about the food/nutrient
            food_match = re.search(r'([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\s*\d', measurement['context'])
            if not food_match:
                continue
                
            food_or_nutrient = food_match.group(1).strip().lower()
            
            # Generate research-focused insight
            insight_text = self.generate_insight(measurement['context'], food_or_nutrient)
            if not insight_text:
                continue
            
            # Extract affected biomarkers
            biomarkers = self.extract_biomarkers(measurement['context'])
            
            # Extract statistical findings
            stats = self.extract_statistical_findings(measurement['context'])
            
            # Determine intervention type
            intervention_type = "diet_modification"
            if "supplement" in measurement['context'].lower():
                intervention_type = "supplementation"
            elif "experiment" in measurement['context'].lower() or "test" in measurement['context'].lower():
                intervention_type = "experiment"
            
            # Calculate confidence based on quality of evidence
            confidence = self._calculate_confidence(measurement['context'], stats, biomarkers)
            
            evidence = DietaryEvidence(
                measurement=measurement['measurement'],
                context=measurement['context'],
                impact=self._extract_impact(measurement['context']),
                source_video=video_title
            )
            
            # Look for existing insight about this food/nutrient
            existing = next((i for i in insights if i.food_or_nutrient == food_or_nutrient), None)
            if existing:
                existing.evidence.append(evidence)
                existing.biomarkers = list(set(existing.biomarkers + biomarkers))
                existing.confidence = max(existing.confidence, confidence)
            else:
                insights.append(DietaryInsight(
                    food_or_nutrient=food_or_nutrient,
                    key_finding=insight_text,
                    evidence=[evidence],
                    biomarkers=biomarkers,
                    confidence=confidence,
                    intervention_type=intervention_type
                ))
        
        return insights

    def _calculate_confidence(self, context: str, stats: List[Dict[str, str]], biomarkers: List[str]) -> float:
        """Calculate confidence score based on evidence quality."""
        confidence = 0.0
        
        # Higher confidence for statistical evidence
        if stats:
            confidence += 0.3
            
        # Higher confidence for multiple biomarkers
        if len(biomarkers) > 1:
            confidence += 0.2
            
        # Higher confidence for specific measurements
        if re.search(r'\d+(?:\.\d+)?(?:\s*%|\s*mg|\s*g|\s*mcg|\s*IU)', context):
            confidence += 0.2
            
        # Higher confidence for experimental evidence
        if re.search(r'experiment|test|trial|study', context, re.IGNORECASE):
            confidence += 0.2
            
        # Base confidence
        confidence += 0.1
        
        return min(confidence, 1.0)

    def _extract_impact(self, context: str) -> Optional[str]:
        """Extract the reported impact from context."""
        impact_patterns = [
            r'(significantly\s+(?:increased|decreased|improved|reduced)\s+[^\.]+)',
            r'(associated\s+with\s+(?:higher|lower|better|worse)\s+[^\.]+)',
            r'(correlated\s+with\s+[^\.]+)'
        ]
        
        for pattern in impact_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

class ParallelTranscriptProcessor:
    """Processes transcripts in parallel to extract insights."""
    
    def __init__(self, num_workers: Optional[int] = None):
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)
        print(f"Initializing with {self.num_workers} workers")
        self.extractor = OllamaInsightExtractor()

    def process_directory(self, transcript_dir: str, output_file: str):
        """Process all transcripts and combine insights."""
        transcript_dir = Path(transcript_dir)
        transcript_files = list(transcript_dir.glob("*.md"))
        print(f"Found {len(transcript_files)} transcript files")
        
        all_insights = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self._process_file, str(f)): f 
                      for f in transcript_files}
            
            for future in tqdm(futures, desc="Processing transcripts"):
                try:
                    file_insights = future.result()
                    if file_insights:
                        all_insights.extend(file_insights)
                except Exception as e:
                    print(f"Error processing {futures[future]}: {str(e)}")
        
        # Combine related insights
        combined_insights = self._combine_related_insights(all_insights)
        
        # Save results
        self._save_insights(combined_insights, output_file)
        self._generate_report(combined_insights, output_file.replace('.json', '_report.md'))

    def _process_file(self, file_path: str) -> List[DietaryInsight]:
        """Process a single transcript file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.extractor.process_transcript(content, Path(file_path).stem)
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
            return []

    def _combine_related_insights(self, insights: List[DietaryInsight]) -> List[DietaryInsight]:
        """Combine related insights about the same food/nutrient."""
        combined = defaultdict(list)
        
        # Group by food/nutrient
        for insight in insights:
            combined[insight.food_or_nutrient].append(insight)
        
        # Merge insights about the same food/nutrient
        result = []
        for food, food_insights in combined.items():
            if len(food_insights) == 1:
                result.append(food_insights[0])
            else:
                # Merge multiple insights
                result.append(DietaryInsight(
                    food_or_nutrient=food,
                    key_finding=self._merge_findings([i.key_finding for i in food_insights]),
                    evidence=sum((i.evidence for i in food_insights), []),
                    biomarkers=list(set(sum((i.biomarkers for i in food_insights), []))),
                    confidence=max(i.confidence for i in food_insights),
                    intervention_type=max(set(i.intervention_type for i in food_insights), 
                                       key=lambda x: sum(1 for i in food_insights if i.intervention_type == x))
                ))
        
        return result

    def _merge_findings(self, findings: List[str]) -> str:
        """Merge multiple findings into a coherent summary."""
        if not findings:
            return ""
        
        # Remove duplicates while preserving order
        findings = list(dict.fromkeys([f.strip() for f in findings if f.strip()]))
        
        # Use Mistral to generate a coherent summary
        combined = " ".join(findings)
        prompt = f"Summarize these related findings into a single coherent insight: {combined}"
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "mistral", "prompt": prompt, "stream": False}
            )
            if response.status_code == 200:
                return response.json().get('response', '').strip()
            return findings[0]  # Fallback to first finding if API fails
        except:
            return findings[0]

    def _save_insights(self, insights: List[DietaryInsight], output_file: str):
        """Save insights to JSON file."""
        # Create directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([i.to_dict() for i in insights], f, indent=2)

    def _generate_report(self, insights: List[DietaryInsight], output_file: str):
        """Generate a detailed research report of the insights."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 🧬 Diet & Longevity Research Insights\n\n")
            
            # Write summary
            f.write("## 📊 Summary\n")
            f.write(f"- Total unique foods/nutrients analyzed: {len(insights)}\n")
            f.write(f"- Total biomarkers affected: {len(set(sum([i.biomarkers for i in insights], [])))}\n")
            f.write(f"- Total pieces of evidence: {sum(len(i.evidence) for i in insights)}\n\n")
            
            # Group by intervention type
            by_type = defaultdict(list)
            for insight in insights:
                by_type[insight.intervention_type].append(insight)
            
            # Write detailed findings
            for int_type, type_insights in sorted(by_type.items()):
                f.write(f"## {int_type.replace('_', ' ').title()}\n\n")
                
                # Sort by confidence
                for insight in sorted(type_insights, key=lambda x: x.confidence, reverse=True):
                    f.write(f"### {insight.food_or_nutrient.title()}\n\n")
                    f.write(f"**Key Finding:** {insight.key_finding}\n\n")
                    f.write(f"**Confidence Score:** {'★' * int(insight.confidence * 5)}\n\n")
                    
                    if insight.biomarkers:
                        f.write("**Affected Biomarkers:**\n")
                        for biomarker in sorted(insight.biomarkers):
                            f.write(f"- {biomarker}\n")
                        f.write("\n")
                    
                    f.write("**Evidence:**\n")
                    for evidence in insight.evidence:
                        f.write(f"- Measurement: {evidence.measurement}\n")
                        if evidence.impact:
                            f.write(f"  Impact: {evidence.impact}\n")
                        f.write(f"  Context: \"{evidence.context}\"\n")
                        f.write(f"  Source: {evidence.source_video}\n\n")
                    f.write("\n---\n\n")


def main():
    """Main function to run the insight extraction process."""
    print("Starting Diet Insights Extraction using Mistral")
    
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/version")
        if response.status_code != 200:
            raise Exception("Ollama is not responding properly")
    except Exception as e:
        print("Error: Ollama is not running. Please start Ollama first.")
        print("Install instructions:")
        print("1. Install Ollama from https://ollama.ai")
        print("2. Run: ollama pull mistral")
        print("3. Start Ollama and try again")
        return
    
    print("Ollama is running and ready")
    
    processor = ParallelTranscriptProcessor()
    
    try:
        processor.process_directory(
            "transcripts",
            "diet_insights.json"
        )
        print("\nProcessing completed successfully!")
        print("\nOutput files:")
        print("- diet_insights.json: Raw data with all insights")
        print("- diet_insights_report.md: Human-readable report")
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        print("Please check the error message above and try again.")


if __name__ == "__main__":
    main()