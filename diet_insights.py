from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Set, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import json
from collections import defaultdict
from tqdm import tqdm
import re
import requests
import time

@dataclass
class DietInsight:
    """Represents a meaningful dietary insight with context and relationships."""
    key_finding: str
    evidence: List[str]
    related_insights: List[str]
    source_videos: List[str]
    confidence: float
    category: str

class OllamaInsightExtractor:
    """Extracts insights using Ollama-powered Mistral."""
    
    def __init__(self, model="mistral"):
        self.model = model
        self.base_url = "http://localhost:11434/api/generate"
        
        # System prompts for different extraction tasks
        self.prompts = {
            'key_finding': """You are an expert in nutrition and dietary science. Extract the key dietary insight from this text. 
            Focus on specific, actionable findings about nutrition, health effects, or mechanisms.
            Think step by step:
            1. Identify the main dietary topic
            2. Find specific claims or findings
            3. Extract any numerical data or specific effects
            4. Formulate a clear, concise insight
            
            Text: {text}
            
            Key insight:""",
            
            'evidence': """Find specific evidence that supports the dietary claim in this text.
            Look for:
            - Research studies or clinical trials
            - Specific numbers or measurements
            - Mechanisms of action
            - Expert citations
            - Real-world examples
            
            Text: {text}
            
            Evidence:""",
            
            'mechanism': """Explain the biological or chemical mechanism described in this text.
            Focus on:
            - Metabolic pathways
            - Nutrient interactions
            - Physiological effects
            - Cellular or molecular processes
            
            Text: {text}
            
            Mechanism:"""
        }

    def generate_insight(self, text: str, prompt_key: str) -> str:
        """Generate an insight using Ollama."""
        prompt = self.prompts[prompt_key].format(text=text)
        
        response = requests.post(
            self.base_url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
        )
        
        if response.status_code == 200:
            return response.json().get('response', '').strip()
        else:
            raise Exception(f"Error from Ollama API: {response.text}")

    def chunk_transcript(self, text: str, chunk_size: int = 2000) -> List[str]:
        """Split transcript into meaningful chunks for processing."""
        # Split on paragraph boundaries or sentence boundaries
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            if current_length + len(para) > chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(para)
            current_length += len(para)
            
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            
        return chunks

    def extract_insights(self, text: str, video_title: str) -> List[DietInsight]:
        """Extract multiple insights from a transcript."""
        insights = []
        chunks = self.chunk_transcript(text)
        
        for chunk in tqdm(chunks, desc=f"Processing {video_title}", leave=False):
            try:
                # Generate key finding
                key_finding = self.generate_insight(chunk, 'key_finding')
                
                # Only proceed if we found a meaningful dietary insight
                if self._is_meaningful_insight(key_finding):
                    # Get supporting evidence
                    evidence = self.generate_insight(chunk, 'evidence')
                    
                    # Try to understand mechanism
                    mechanism = self.generate_insight(chunk, 'mechanism')
                    
                    # Add a small delay to avoid overwhelming Ollama
                    time.sleep(0.1)
                    
                    insight = DietInsight(
                        key_finding=key_finding,
                        evidence=[evidence] if evidence else [],
                        related_insights=[mechanism] if mechanism else [],
                        source_videos=[video_title],
                        confidence=self._calculate_confidence(key_finding, evidence),
                        category=self._categorize_insight(key_finding)
                    )
                    insights.append(insight)
            except Exception as e:
                print(f"Error processing chunk from {video_title}: {str(e)}")
                continue
        
        return insights

    def _is_meaningful_insight(self, finding: str) -> bool:
        """Determine if an extracted finding is meaningful."""
        if not finding or len(finding) < 20:
            return False
            
        # Look for specific dietary terms
        dietary_terms = {
            'nutrient', 'vitamin', 'mineral', 'protein', 'fat', 'carbohydrate',
            'diet', 'food', 'supplement', 'meal', 'nutrition', 'metabolic',
            'blood', 'health', 'effect', 'impact', 'increase', 'decrease',
            'improve', 'reduce', 'benefit', 'risk'
        }
        
        words = set(finding.lower().split())
        if not any(term in words for term in dietary_terms):
            return False
            
        return True

    def _calculate_confidence(self, finding: str, evidence: str) -> float:
        """Calculate confidence score based on specificity and evidence."""
        confidence = 0.0
        
        # Higher confidence for specific numbers/measurements
        if re.search(r'\d+(?:\.\d+)?(?:\s*%|\s*mg|\s*g|\s*mcg|\s*IU)', finding):
            confidence += 0.3
            
        # Higher confidence for referenced studies
        if re.search(r'study|research|trial|evidence|published|journal', evidence, re.IGNORECASE):
            confidence += 0.3
            
        # Higher confidence for explained mechanisms
        if re.search(r'because|through|by|via|mechanism|pathway|process', evidence, re.IGNORECASE):
            confidence += 0.2
            
        # Higher confidence for specific biomarkers or measurements
        if re.search(r'blood|serum|plasma|level|concentration|biomarker', evidence, re.IGNORECASE):
            confidence += 0.2
            
        # Base confidence
        confidence += 0.2
        
        return min(confidence, 1.0)

    def _categorize_insight(self, finding: str) -> str:
        """Categorize the type of insight."""
        categories = {
            'recommendation': r'should|recommend|advise|best|optimal|consider|try',
            'mechanism': r'because|cause|through|pathway|mechanism|process|via',
            'observation': r'found|observed|showed|demonstrated|reveals|indicates',
            'comparison': r'compared|versus|better|worse|than|higher|lower',
            'correlation': r'associated|linked|correlation|relationship|connection',
            'dosage': r'dose|amount|serving|quantity|level|concentration',
            'timing': r'timing|when|time|duration|period|frequency|daily'
        }
        
        for category, pattern in categories.items():
            if re.search(pattern, finding, re.IGNORECASE):
                return category
        return 'general'

class ParallelInsightProcessor:
    """Processes transcripts in parallel to extract insights."""
    
    def __init__(self, num_workers: Optional[int] = None):
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)
        print(f"Initializing with {self.num_workers} workers")
        self.extractor = OllamaInsightExtractor()

    def process_directory(self, transcript_dir: str, output_file: str):
        """Process all transcripts in parallel and combine insights."""
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

        print(f"Extracted {len(all_insights)} initial insights")
        
        # Combine related insights
        combined_insights = self._combine_related_insights(all_insights)
        print(f"Combined into {len(combined_insights)} unique insights")
        
        # Save results
        self._save_insights(combined_insights, output_file)
        self._generate_report(combined_insights, output_file.replace('.json', '_report.md'))

    def _process_file(self, file_path: str) -> List[DietInsight]:
        """Process a single transcript file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.extractor.extract_insights(content, Path(file_path).stem)
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
            return []

    def _combine_related_insights(self, insights: List[DietInsight]) -> List[DietInsight]:
        """Combine insights that are semantically related."""
        combined = []
        merged = set()
        
        for i, insight1 in enumerate(insights):
            if i in merged:
                continue
                
            related = []
            for j, insight2 in enumerate(insights[i+1:], i+1):
                if j not in merged and self._are_insights_related(insight1, insight2):
                    related.append(insight2)
                    merged.add(j)
            
            if related:
                combined.append(self._merge_insights(insight1, related))
            else:
                combined.append(insight1)
        
        return combined

    def _are_insights_related(self, insight1: DietInsight, insight2: DietInsight) -> bool:
        """Determine if two insights are semantically related."""
        # Improved similarity check using key terms
        terms1 = set(re.findall(r'\b\w+\b', insight1.key_finding.lower()))
        terms2 = set(re.findall(r'\b\w+\b', insight2.key_finding.lower()))
        
        # Calculate Jaccard similarity
        intersection = len(terms1 & terms2)
        union = len(terms1 | terms2)
        
        if union == 0:
            return False
            
        similarity = intersection / union
        return similarity > 0.3

    def _merge_insights(self, main_insight: DietInsight, related: List[DietInsight]) -> DietInsight:
        """Merge related insights into a single, comprehensive insight."""
        return DietInsight(
            key_finding=main_insight.key_finding,
            evidence=list(set(main_insight.evidence + sum((i.evidence for i in related), []))),
            related_insights=list(set(main_insight.related_insights + sum((i.related_insights for i in related), []))),
            source_videos=list(set(main_insight.source_videos + sum((i.source_videos for i in related), []))),
            confidence=max(main_insight.confidence, *(i.confidence for i in related)),
            category=main_insight.category
        )

    def _save_insights(self, insights: List[DietInsight], output_file: str):
        """Save insights to JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([vars(i) for i in insights], f, indent=2)

    def _generate_report(self, insights: List[DietInsight], output_file: str):
        """Generate a readable report of the insights."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 🍎 Dietary Insights Report\n\n")
            
            # Group insights by category
            by_category = defaultdict(list)
            for insight in insights:
                by_category[insight.category].append(insight)
            
            # Write summary
            f.write("## 📊 Summary\n")
            f.write(f"- Total unique insights: {len(insights)}\n")
            f.write(f"- Categories covered: {len(by_category)}\n")
            f.write(f"- Total source videos: {len(set(sum([i.source_videos for i in insights], [])))}\n\n")
            
            # Write insights by category
            for category, category_insights in sorted(by_category.items()):
                f.write(f"## {category.title()} Insights\n\n")
                
                # Sort by confidence
                for insight in sorted(category_insights, key=lambda x: x.confidence, reverse=True):
                    f.write(f"### {insight.key_finding}\n")
                    f.write(f"**Confidence:** {'★' * int(insight.confidence * 5)}\n\n")
                    
                    if insight.evidence:
                        f.write("**Evidence:**\n")
                        for evidence in insight.evidence:
                            f.write(f"- {evidence}\n")
                        f.write("\n")
                    
                    if insight.related_insights:
                        f.write("**Related Insights:**\n")
                        for related in insight.related_insights:
                            f.write(f"- {related}\n")
                        f.write("\n")
                    
                    f.write("**Source Videos:**\n")
                    for source in sorted(insight.source_videos):
                        f.write(f"- {source}\n")
                    f.write("\n")

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
    
    processor = ParallelInsightProcessor()
    
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