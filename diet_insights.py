from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import json
from collections import defaultdict
import re
import requests
import time
from datetime import datetime

@dataclass
class BiomarkerInsight:
    """Represents a relationship between a dietary factor and a biomarker."""
    biomarker: str
    direction: str  # "increases" or "decreases"
    magnitude: float  # correlation strength
    confidence: float
    supporting_evidence: List[str]
    context: str
    age_related_change: str  # how this biomarker changes with aging

@dataclass
class DietaryMechanism:
    """Represents a biological mechanism linking diet to health outcomes."""
    pathway: str
    inputs: List[str]  # dietary factors
    outputs: List[str]  # affected biomarkers/systems
    confidence: float
    evidence: List[str]
    source_videos: List[str]

@dataclass
class StudyReference:
    """Represents a referenced scientific study."""
    description: str
    population_size: Optional[int]
    finding: str
    confidence: float
    source_video: str
    timestamp: Optional[str]

@dataclass
class DietaryInsight:
    """Enhanced insight representation capturing complex relationships."""
    key_finding: str
    mechanisms: List[DietaryMechanism]
    biomarker_effects: List[BiomarkerInsight]
    referenced_studies: List[StudyReference]
    contradictions: List[str]  # captures contradictory findings
    limitations: List[str]  # captures caveats and limitations
    practical_implications: List[str]  # actionable takeaways
    confidence_score: float
    source_videos: List[str]
    context: str
    category: str
    interconnections: List[str]  # connections to other insights

class InsightPromptGenerator:
    """Generates sophisticated prompts for Mistral."""
    
    def __init__(self):
        self.biomarker_patterns = {
            'glucose': r'glucose|blood sugar|glycemia|hba1c',
            'lipids': r'cholesterol|triglycerides|hdl|ldl|vldl',
            'inflammation': r'crp|c-reactive|inflammation|inflammatory',
            'hormones': r'testosterone|dhea|cortisol|insulin',
            'aging': r'biological age|epigenetic|methylation|telomere',
            'liver': r'alt|ast|ggt|liver enzymes|albumin',
            'kidney': r'creatinine|egfr|bun|uric acid',
            'blood': r'hemoglobin|hematocrit|rbc|wbc|platelets'
        }
        
        self.mechanism_patterns = {
            'metabolic': r'metabolism|energy|atp|mitochondria',
            'signaling': r'pathway|cascade|receptor|signaling',
            'genetic': r'gene|expression|transcription|methylation',
            'immune': r'immune|inflammation|cytokine|interleukin',
            'oxidative': r'oxidative stress|ros|antioxidant|radical'
        }

    def generate_prompts(self, text: str) -> List[str]:
        """Generates focused prompts based on content analysis."""
        prompts = []
        
        # Biomarker analysis prompt
        for system, pattern in self.biomarker_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                prompts.append(self._generate_biomarker_prompt(system, text))
        
        # Mechanism analysis prompt
        for pathway, pattern in self.mechanism_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                prompts.append(self._generate_mechanism_prompt(pathway, text))
        
        # Study analysis prompt
        if re.search(r'study|research|trial|paper|published', text, re.IGNORECASE):
            prompts.append(self._generate_study_prompt(text))
        
        # Correlation analysis prompt
        if re.search(r'correlation|associated|relationship|linked', text, re.IGNORECASE):
            prompts.append(self._generate_correlation_prompt(text))
        
        return prompts

    def _generate_biomarker_prompt(self, system: str, text: str) -> str:
        return f"""Analyze the relationship between dietary factors and {system} biomarkers in this text.
        Focus on:
        1. Direct effects on biomarker levels
        2. Age-related changes in these biomarkers
        3. Mechanisms of action
        4. Practical implications for diet
        
        Text: {text}
        
        Analysis:"""

    def _generate_mechanism_prompt(self, pathway: str, text: str) -> str:
        return f"""Explain the {pathway} mechanisms described in this text.
        Include:
        1. Key pathways and processes
        2. Dietary triggers and modulators
        3. Downstream effects on health
        4. Time dynamics and dose relationships
        
        Text: {text}
        
        Mechanism:"""

    def _generate_study_prompt(self, text: str) -> str:
        return f"""Extract key study findings from this text.
        Include:
        1. Population size and characteristics
        2. Key findings and effect sizes
        3. Methodology strengths/limitations
        4. Practical implications
        
        Text: {text}
        
        Findings:"""

    def _generate_correlation_prompt(self, text: str) -> str:
        return f"""Analyze the correlations discussed in this text.
        Focus on:
        1. Direction and strength of relationships
        2. Confounding factors considered
        3. Temporal relationships
        4. Causation vs correlation distinction
        
        Text: {text}
        
        Analysis:"""

class OllamaInsightExtractor:
    """Enhanced insight extractor using Ollama."""
    
    def __init__(self, model="mistral"):
        self.model = model
        self.base_url = "http://localhost:11434/api/generate"
        self.prompt_generator = InsightPromptGenerator()
        
        # Track processed insights for cross-referencing
        self.insight_registry = defaultdict(list)
        
        # Define dietary categories for classification
        self.categories = {
            'nutrient_optimization': {
                'patterns': ['vitamin', 'mineral', 'nutrient', 'deficiency', 'optimization'],
                'importance': 'fundamental'
            },
            'metabolic_health': {
                'patterns': ['glucose', 'insulin', 'metabolism', 'energy'],
                'importance': 'critical'
            },
            'inflammation': {
                'patterns': ['inflammation', 'immune', 'cytokine', 'oxidative'],
                'importance': 'major'
            },
            'longevity': {
                'patterns': ['aging', 'longevity', 'lifespan', 'epigenetic'],
                'importance': 'strategic'
            },
            'biomarker_optimization': {
                'patterns': ['biomarker', 'blood test', 'levels', 'optimize'],
                'importance': 'tactical'
            }
        }

    def generate_insight(self, text: str, prompt: str) -> str:
        """Generate an insight using Ollama API."""
        try:
            response = requests.post(
                self.base_url,
                json={
                    "model": self.model,
                    "prompt": f"{prompt}\n\nContext: {text}",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                },
                timeout=30  # Add timeout to avoid hanging
            )
            
            if response.status_code == 200:
                return response.json()["response"].strip()
            else:
                print(f"Error from Ollama API: {response.text}")
                return ""
                
        except requests.exceptions.Timeout:
            print("Timeout while generating insight")
            return ""
        except Exception as e:
            print(f"Error generating insight: {str(e)}")
            return ""

    def extract_insights(self, text: str, video_title: str) -> List[DietaryInsight]:
        """Extract deep insights from transcript text."""
        insights = []
        chunks = self._smart_chunk_transcript(text)
        
        for chunk in chunks:
            # Generate focused prompts based on content
            prompts = self.prompt_generator.generate_prompts(chunk)
            
            chunk_insights = []
            for prompt in prompts:
                try:
                    # Get raw insight from Mistral
                    raw_insight = self.generate_insight(chunk, prompt)
                    
                    # Parse and structure the insight
                    structured_insight = self._structure_insight(raw_insight, chunk, video_title)
                    
                    if structured_insight and self._is_high_quality_insight(structured_insight):
                        chunk_insights.append(structured_insight)
                        
                except Exception as e:
                    print(f"Error processing chunk from {video_title}: {str(e)}")
                    continue
            
            # Combine related insights from same chunk
            if chunk_insights:
                combined_insight = self._combine_chunk_insights(chunk_insights)
                insights.append(combined_insight)
                
                # Update registry for cross-referencing
                self._update_insight_registry(combined_insight)
        
        # Find connections between insights
        insights = self._cross_reference_insights(insights)
        
        return insights

    def _smart_chunk_transcript(self, text: str) -> List[str]:
        """Intelligently chunk transcript preserving context."""
        # Split on topic transitions first
        topic_chunks = re.split(r'\n(?=(?:[A-Z][^.!?]*)(?:[.!?]+\s|\n))', text)
        
        processed_chunks = []
        for chunk in topic_chunks:
            # Further split if chunk is too large
            if len(chunk) > 2000:
                # Split on paragraph boundaries
                sub_chunks = chunk.split('\n\n')
                current_chunk = []
                current_length = 0
                
                for sub_chunk in sub_chunks:
                    if current_length + len(sub_chunk) > 2000 and current_chunk:
                        processed_chunks.append('\n\n'.join(current_chunk))
                        current_chunk = []
                        current_length = 0
                    current_chunk.append(sub_chunk)
                    current_length += len(sub_chunk)
                
                if current_chunk:
                    processed_chunks.append('\n\n'.join(current_chunk))
            else:
                processed_chunks.append(chunk)
        
        return processed_chunks

    def _structure_insight(self, raw_insight: str, context: str, video_title: str) -> Optional[DietaryInsight]:
        """Structure raw insight into comprehensive DietaryInsight object."""
        try:
            # Extract mechanisms
            mechanisms = self._extract_mechanisms(raw_insight, context)
            
            # Extract biomarker effects
            biomarker_effects = self._extract_biomarker_effects(raw_insight, context)
            
            # Extract study references
            studies = self._extract_study_references(raw_insight, video_title)
            
            # Extract contradictions and limitations
            contradictions = self._extract_contradictions(raw_insight)
            limitations = self._extract_limitations(raw_insight)
            
            # Generate practical implications
            implications = self._generate_implications(raw_insight, mechanisms, biomarker_effects)
            
            # Calculate confidence score
            confidence = self._calculate_insight_confidence(
                mechanisms, biomarker_effects, studies, contradictions, limitations
            )
            
            # Determine category
            category = self._categorize_insight(raw_insight, mechanisms, biomarker_effects)
            
            return DietaryInsight(
                key_finding=self._extract_key_finding(raw_insight),
                mechanisms=mechanisms,
                biomarker_effects=biomarker_effects,
                referenced_studies=studies,
                contradictions=contradictions,
                limitations=limitations,
                practical_implications=implications,
                confidence_score=confidence,
                source_videos=[video_title],
                context=context,
                category=category,
                interconnections=[]  # Will be filled during cross-referencing
            )
            
        except Exception as e:
            print(f"Error structuring insight: {str(e)}")
            return None

    def _extract_mechanisms(self, text: str, context: str) -> List[DietaryMechanism]:
        """Extract biological mechanisms from text."""
        mechanisms = []
        
        # Identify mechanism descriptions
        mechanism_patterns = [
            r'(?:through|via|by|because|due to)\s+([^.]+)',
            r'mechanism\s+(?:involves|includes|is)\s+([^.]+)',
            r'pathway[s]?\s+(?:involves|includes|is)\s+([^.]+)'
        ]
        
        for pattern in mechanism_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                mechanism_text = match.group(1)
                
                # Extract inputs (dietary factors)
                inputs = re.findall(r'(?:intake of|consuming|supplementing with)\s+([^,]+)', mechanism_text)
                
                # Extract outputs (affected systems/biomarkers)
                outputs = re.findall(r'(?:affects|impacts|changes|modifies)\s+([^,]+)', mechanism_text)
                
                # Calculate confidence
                confidence = self._calculate_mechanism_confidence(mechanism_text, context)
                
                mechanisms.append(DietaryMechanism(
                    pathway=mechanism_text.strip(),
                    inputs=inputs if inputs else [],
                    outputs=outputs if outputs else [],
                    confidence=confidence,
                    evidence=[context],  # Use context as evidence
                    source_videos=[]  # Will be filled later
                ))
        
        return mechanisms

    def _extract_biomarker_effects(self, text: str, context: str) -> List[BiomarkerInsight]:
        """Extract effects on biomarkers from text."""
        biomarker_effects = []
        
        # Pattern for biomarker changes
        effect_pattern = r'(increases?|decreases?|reduces?|lowers?|elevates?|improves?)\s+([^,.]+)'
        
        matches = re.finditer(effect_pattern, text, re.IGNORECASE)
        for match in matches:
            direction = match.group(1)
            biomarker = match.group(2)
            
            # Calculate magnitude and confidence
            magnitude = self._estimate_effect_magnitude(text, biomarker)
            confidence = self._calculate_biomarker_confidence(text, biomarker, context)
            
            # Determine age-related change
            age_change = self._determine_age_related_change(biomarker)
            
            biomarker_effects.append(BiomarkerInsight(
                biomarker=biomarker.strip(),
                direction="increases" if "increase" in direction.lower() else "decreases",
                magnitude=magnitude,
                confidence=confidence,
                supporting_evidence=[context],
                context=context,
                age_related_change=age_change
            ))
        
        return biomarker_effects

    def _extract_study_references(self, text: str, video_title: str) -> List[StudyReference]:
        """Extract referenced studies from text."""
        studies = []
        
        # Pattern for study mentions
        study_pattern = r'(?:study|research|trial|paper)\s+(?:of|with|in|showed|found)\s+([^.]+)'
        
        matches = re.finditer(study_pattern, text, re.IGNORECASE)
        for match in matches:
            study_text = match.group(1)
            
            # Extract population size if mentioned
            pop_size = None
            pop_match = re.search(r'(\d+)\s+(?:people|subjects|participants)', study_text)
            if pop_match:
                pop_size = int(pop_match.group(1))
            
            # Calculate confidence
            confidence = self._calculate_study_confidence(study_text)
            
            studies.append(StudyReference(
                description=study_text.strip(),
                population_size=pop_size,
                finding=self._extract_study_finding(study_text),
                confidence=confidence,
                source_video=video_title,
                timestamp=self._extract_timestamp(study_text)
            ))
        
        return studies

    def _extract_contradictions(self, text: str) -> List[str]:
        """Extract contradictory findings or caveats."""
        contradictions = []
        
        contradiction_patterns = [
            r'however[,\s]+([^.]+)',
            r'but[,\s]+([^.]+)',
            r'on the other hand[,\s]+([^.]+)',
            r'in contrast[,\s]+([^.]+)',
            r'surprisingly[,\s]+([^.]+)',
            r'unexpectedly[,\s]+([^.]+)'
        ]
        
        for pattern in contradiction_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                contradictions.append(match.group(1).strip())
        
        return contradictions

    def _extract_limitations(self, text: str) -> List[str]:
        """Extract study limitations and caveats."""
        limitations = []
        
        limitation_patterns = [
            r'limitation[s]?[,\s]+(?:is|are|includes?)[,\s]+([^.]+)',
            r'(?:note|notably)[,\s]+([^.]+)',
            r'caveat[s]?[,\s]+([^.]+)',
            r'(?:may|might|could)[,\s]+([^.]+)',
            r'further research[,\s]+([^.]+)'
        ]
        
        for pattern in limitation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                limitations.append(match.group(1).strip())
        
        return limitations

    def _generate_implications(
        self, text: str, 
        mechanisms: List[DietaryMechanism], 
        biomarker_effects: List[BiomarkerInsight]
    ) -> List[str]:
        """Generate practical implications from insights."""
        implications = []
        
        # Pattern for direct recommendations
        rec_patterns = [
            r'(?:should|recommend|advised?|optimal)\s+([^.]+)',
            r'(?:increase|decrease|reduce|improve)\s+intake[^.]+',
            r'(?:beneficial|important|crucial|essential)\s+to[^.]+',
            r'(?:target|goal|aim)[^.]+(?:is|should be)[^.]+'
        ]
        
        for pattern in rec_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                implications.append(match.group(0).strip())
        
        # Add implications from mechanisms
        for mechanism in mechanisms:
            if mechanism.confidence > 0.7:  # Only use high-confidence mechanisms
                for input_factor in mechanism.inputs:
                    implications.append(
                        f"Consider {input_factor} to influence {', '.join(mechanism.outputs)}"
                    )
        
        # Add implications from biomarker effects
        for effect in biomarker_effects:
            if effect.confidence > 0.7:  # Only use high-confidence effects
                implications.append(
                    f"Monitor {effect.biomarker} as it {effect.direction} with this intervention"
                )
        
        return list(set(implications))  # Remove duplicates

    def _calculate_insight_confidence(
        self,
        mechanisms: List[DietaryMechanism],
        biomarker_effects: List[BiomarkerInsight],
        studies: List[StudyReference],
        contradictions: List[str],
        limitations: List[str]
    ) -> float:
        """Calculate overall confidence score for an insight."""
        confidence = 0.5  # Start with neutral confidence
        
        # Add confidence from mechanisms
        if mechanisms:
            mech_conf = sum(m.confidence for m in mechanisms) / len(mechanisms)
            confidence += 0.1 * mech_conf
        
        # Add confidence from biomarker effects
        if biomarker_effects:
            bio_conf = sum(b.confidence for b in biomarker_effects) / len(biomarker_effects)
            confidence += 0.1 * bio_conf
        
        # Add confidence from studies
        if studies:
            study_conf = sum(s.confidence for s in studies) / len(studies)
            confidence += 0.2 * study_conf
            
            # Bonus for large studies
            large_studies = sum(1 for s in studies if s.population_size and s.population_size > 1000)
            if large_studies:
                confidence += 0.1 * (large_studies / len(studies))
        
        # Reduce confidence for contradictions and limitations
        confidence -= 0.05 * len(contradictions)
        confidence -= 0.05 * len(limitations)
        
        return min(max(confidence, 0.0), 1.0)  # Ensure between 0 and 1

    def _categorize_insight(
        self, 
        text: str,
        mechanisms: List[DietaryMechanism],
        biomarker_effects: List[BiomarkerInsight]
    ) -> str:
        """Categorize insight based on content analysis."""
        # Count category-related terms
        category_scores = defaultdict(float)
        
        for category, info in self.categories.items():
            base_score = 0
            
            # Check patterns in text
            for pattern in info['patterns']:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                base_score += sum(1 for _ in matches)
            
            # Add scores from mechanisms
            for mechanism in mechanisms:
                for pattern in info['patterns']:
                    if re.search(pattern, mechanism.pathway, re.IGNORECASE):
                        base_score += 0.5 * mechanism.confidence
            
            # Add scores from biomarker effects
            for effect in biomarker_effects:
                for pattern in info['patterns']:
                    if re.search(pattern, effect.biomarker, re.IGNORECASE):
                        base_score += 0.5 * effect.confidence
            
            # Weight by importance
            importance_weights = {
                'fundamental': 1.0,
                'critical': 0.9,
                'major': 0.8,
                'strategic': 0.7,
                'tactical': 0.6
            }
            
            category_scores[category] = base_score * importance_weights[info['importance']]
        
        # Return category with highest score, or 'general' if no strong match
        if category_scores:
            max_category = max(category_scores.items(), key=lambda x: x[1])
            if max_category[1] > 0:
                return max_category[0]
        
        return 'general'

    def _update_insight_registry(self, insight: DietaryInsight):
        """Update registry with new insight for cross-referencing."""
        # Register by biomarkers
        for effect in insight.biomarker_effects:
            self.insight_registry[f"biomarker:{effect.biomarker}"].append(insight)
        
        # Register by mechanisms
        for mechanism in insight.mechanisms:
            self.insight_registry[f"mechanism:{mechanism.pathway}"].append(insight)
        
        # Register by category
        self.insight_registry[f"category:{insight.category}"].append(insight)
        
        # Register key terms
        key_terms = re.findall(r'\b\w+\b', insight.key_finding.lower())
        for term in key_terms:
            self.insight_registry[f"term:{term}"].append(insight)

    def _cross_reference_insights(self, insights: List[DietaryInsight]) -> List[DietaryInsight]:
        """Find and add connections between insights."""
        for insight in insights:
            connections = set()
            
            # Check for biomarker connections
            for effect in insight.biomarker_effects:
                related_insights = self.insight_registry.get(f"biomarker:{effect.biomarker}", [])
                for related in related_insights:
                    if related != insight:
                        connections.add(
                            f"Related to {related.key_finding} through {effect.biomarker}"
                        )
            
            # Check for mechanism connections
            for mechanism in insight.mechanisms:
                related_insights = self.insight_registry.get(f"mechanism:{mechanism.pathway}", [])
                for related in related_insights:
                    if related != insight:
                        connections.add(
                            f"Shares mechanism '{mechanism.pathway}' with: {related.key_finding}"
                        )
            
            # Check for category connections
            category_insights = self.insight_registry.get(f"category:{insight.category}", [])
            for related in category_insights:
                if related != insight:
                    connections.add(
                        f"Related finding in {insight.category}: {related.key_finding}"
                    )
            
            insight.interconnections = list(connections)
        
        return insights

    def _extract_timestamp(self, text: str) -> Optional[str]:
        """Extract timestamp from text if present."""
        timestamp_pattern = r'\[(\d{2}:\d{2}(?::\d{2})?)\]'
        match = re.search(timestamp_pattern, text)
        return match.group(1) if match else None

    def _extract_study_finding(self, text: str) -> str:
        """Extract the main finding from study text."""
        finding_patterns = [
            r'(?:found|showed|demonstrated|revealed|indicated)\s+that\s+([^.]+)',
            r'(?:concluded|suggested|reported)\s+(?:that\s+)?([^.]+)'
        ]
        
        for pattern in finding_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return text.strip()

    def _calculate_mechanism_confidence(self, mechanism_text: str, context: str) -> float:
        """Calculate confidence score for a mechanism."""
        confidence = 0.5  # Start with neutral confidence
        
        # Higher confidence for well-explained mechanisms
        if re.search(r'(?:through|via|by|because|pathway|cascade)', mechanism_text, re.IGNORECASE):
            confidence += 0.2
        
        # Higher confidence for specific molecular/cellular details
        if re.search(r'(?:protein|enzyme|receptor|gene|cell|molecule)', mechanism_text, re.IGNORECASE):
            confidence += 0.2
        
        # Higher confidence for quantified effects
        if re.search(r'\d+(?:\.\d+)?(?:\s*%|\s*fold|\s*times)', mechanism_text, re.IGNORECASE):
            confidence += 0.1
        
        # Reduce confidence for uncertainty language
        if re.search(r'(?:may|might|could|possibly|potentially)', mechanism_text, re.IGNORECASE):
            confidence -= 0.2
        
        return min(max(confidence, 0.0), 1.0)

    def _calculate_biomarker_confidence(self, text: str, biomarker: str, context: str) -> float:
        """Calculate confidence score for a biomarker effect."""
        confidence = 0.5
        
        # Higher confidence for quantified effects
        if re.search(r'\d+(?:\.\d+)?(?:\s*%|\s*units?|\s*points?)', text, re.IGNORECASE):
            confidence += 0.2
        
        # Higher confidence for time-course information
        if re.search(r'(?:after|within|hours?|days?|weeks?|months?)', text, re.IGNORECASE):
            confidence += 0.1
        
        # Higher confidence for dose-response relationships
        if re.search(r'(?:dose|amount|quantity|concentration)', text, re.IGNORECASE):
            confidence += 0.1
        
        # Higher confidence for mechanistic explanations
        if re.search(r'(?:through|via|by|because|mechanism)', text, re.IGNORECASE):
            confidence += 0.1
        
        return min(max(confidence, 0.0), 1.0)

    def _calculate_study_confidence(self, study_text: str) -> float:
        """Calculate confidence score for a study reference."""
        confidence = 0.5
        
        # Higher confidence for larger sample sizes
        sample_match = re.search(r'(\d+)\s+(?:people|subjects|participants)', study_text)
        if sample_match:
            n = int(sample_match.group(1))
            if n > 1000:
                confidence += 0.3
            elif n > 100:
                confidence += 0.2
            else:
                confidence += 0.1
        
        # Higher confidence for controlled studies
        if re.search(r'(?:controlled|randomized|placebo|double-blind)', study_text, re.IGNORECASE):
            confidence += 0.2
        
        # Higher confidence for long-term studies
        if re.search(r'(?:year|month|long-term|longitudinal)', study_text, re.IGNORECASE):
            confidence += 0.1
        
        # Higher confidence for replicated findings
        if re.search(r'(?:replicated|confirmed|validated|consistent)', study_text, re.IGNORECASE):
            confidence += 0.2
        
        return min(max(confidence, 0.0), 1.0)

    def _estimate_effect_magnitude(self, text: str, biomarker: str) -> float:
        """Estimate the magnitude of a biomarker effect."""
        # Look for percentage changes
        pct_match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
        if pct_match:
            return float(pct_match.group(1)) / 100
        
        # Look for fold changes
        fold_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:fold|times)', text)
        if fold_match:
            return float(fold_match.group(1))
        
        # Look for directional language
        if re.search(r'(?:significantly|substantially|markedly)', text, re.IGNORECASE):
            return 0.5
        elif re.search(r'(?:slightly|marginally|mildly)', text, re.IGNORECASE):
            return 0.2
        
        return 0.3  # Default moderate effect

    def _determine_age_related_change(self, biomarker: str) -> str:
        """Determine how a biomarker typically changes with aging."""
        increases_with_age = {
            'glucose', 'insulin', 'inflammation', 'crp', 'blood pressure', 
            'homocysteine', 'triglycerides', 'body fat', 'cortisol'
        }
        
        decreases_with_age = {
            'testosterone', 'dhea', 'growth hormone', 'nad', 'muscle mass',
            'bone density', 'collagen', 'albumin'
        }
        
        biomarker = biomarker.lower()
        
        for term in increases_with_age:
            if term in biomarker:
                return "increases"
                
        for term in decreases_with_age:
            if term in biomarker:
                return "decreases"
                
        return "unknown"

class ParallelInsightProcessor:
    """Enhanced parallel processor for transcript analysis."""
    
    def __init__(self, num_workers: Optional[int] = None):
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)
        print(f"Initializing with {self.num_workers} workers")
        self.extractor = OllamaInsightExtractor()

    def process_directory(self, transcript_dir: str, output_file: str):
        """Process all transcripts and generate comprehensive reports."""
        transcript_dir = Path(transcript_dir)
        transcript_files = list(transcript_dir.glob("*.md"))
        print(f"Found {len(transcript_files)} transcript files")
        
        # Process files in parallel
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
        
        # Generate outputs
        self._save_insights(combined_insights, output_file)
        self._generate_comprehensive_report(combined_insights, output_file.replace('.json', '_report.md'))
        self._generate_network_graph(combined_insights, output_file.replace('.json', '_network.json'))
        self._generate_evidence_matrix(combined_insights, output_file.replace('.json', '_evidence.json'))

    def _process_file(self, file_path: str) -> List[DietaryInsight]:
        """Process a single transcript file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.extractor.extract_insights(content, Path(file_path).stem)
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
            return []

    def _combine_related_insights(self, insights: List[DietaryInsight]) -> List[DietaryInsight]:
        """Combine related insights using enhanced similarity metrics."""
        combined = []
        merged = set()
        
        for i, insight1 in enumerate(insights):
            if i in merged:
                continue
            
            related = []
            for j, insight2 in enumerate(insights[i+1:], i+1):
                if j not in merged and self._are_insights_deeply_related(insight1, insight2):
                    related.append(insight2)
                    merged.add(j)
            
            if related:
                combined.append(self._merge_related_insights(insight1, related))
            else:
                combined.append(insight1)
        
        return combined

    def _are_insights_deeply_related(self, insight1: DietaryInsight, insight2: DietaryInsight) -> bool:
        """Determine if insights are related using multiple criteria."""
        # Check biomarker overlap
        biomarkers1 = {effect.biomarker for effect in insight1.biomarker_effects}
        biomarkers2 = {effect.biomarker for effect in insight2.biomarker_effects}
        biomarker_similarity = len(biomarkers1 & biomarkers2) / len(biomarkers1 | biomarkers2) if biomarkers1 | biomarkers2 else 0
        
        # Check mechanism overlap
        mechanisms1 = {m.pathway for m in insight1.mechanisms}
        mechanisms2 = {m.pathway for m in insight2.mechanisms}
        mechanism_similarity = len(mechanisms1 & mechanisms2) / len(mechanisms1 | mechanisms2) if mechanisms1 | mechanisms2 else 0
        
        # Check key finding similarity using word overlap
        words1 = set(re.findall(r'\b\w+\b', insight1.key_finding.lower()))
        words2 = set(re.findall(r'\b\w+\b', insight2.key_finding.lower()))
        text_similarity = len(words1 & words2) / len(words1 | words2) if words1 | words2 else 0
        
        # Weight and combine similarities
        total_similarity = (
            0.4 * biomarker_similarity + 
            0.4 * mechanism_similarity + 
            0.2 * text_similarity
        )
        
        return total_similarity > 0.3

    def _merge_related_insights(self, main_insight: DietaryInsight, related: List[DietaryInsight]) -> DietaryInsight:
        """Merge related insights while preserving unique information."""
        # Combine mechanisms while removing duplicates
        mechanisms = self._merge_mechanisms(
            [main_insight.mechanisms] + [r.mechanisms for r in related]
        )
        
        # Combine biomarker effects
        biomarker_effects = self._merge_biomarker_effects(
            [main_insight.biomarker_effects] + [r.biomarker_effects for r in related]
        )
        
        # Combine studies
        studies = list({
            (s.description, s.finding): s 
            for s in (main_insight.referenced_studies + sum([r.referenced_studies for r in related], []))
        }.values())
        
        # Combine unique contradictions and limitations
        contradictions = list(set(
            main_insight.contradictions + sum([r.contradictions for r in related], [])
        ))
        limitations = list(set(
            main_insight.limitations + sum([r.limitations for r in related], [])
        ))
        
        # Generate new practical implications considering all merged data
        implications = list(set(
            main_insight.practical_implications + sum([r.practical_implications for r in related], [])
        ))
        
        # Calculate new confidence score
        confidence = self._calculate_merged_confidence(main_insight, related)
        
        return DietaryInsight(
            key_finding=main_insight.key_finding,  # Keep main finding
            mechanisms=mechanisms,
            biomarker_effects=biomarker_effects,
            referenced_studies=studies,
            contradictions=contradictions,
            limitations=limitations,
            practical_implications=implications,
            confidence_score=confidence,
            source_videos=list(set(
                main_insight.source_videos + sum([r.source_videos for r in related], [])
            )),
            context=main_insight.context,  # Keep main context
            category=main_insight.category,  # Keep main category
            interconnections=list(set(
                main_insight.interconnections + sum([r.interconnections for r in related], [])
            ))
        )

    def _merge_mechanisms(self, mechanism_lists: List[List[DietaryMechanism]]) -> List[DietaryMechanism]:
        """Merge mechanisms while preserving unique information."""
        merged = {}
        for mechanisms in mechanism_lists:
            for mech in mechanisms:
                key = mech.pathway
                if key not in merged:
                    merged[key] = mech
                else:
                    # Update existing mechanism with new information
                    existing = merged[key]
                    merged[key] = DietaryMechanism(
                        pathway=existing.pathway,
                        inputs=list(set(existing.inputs + mech.inputs)),
                        outputs=list(set(existing.outputs + mech.outputs)),
                        confidence=max(existing.confidence, mech.confidence),
                        evidence=list(set(existing.evidence + mech.evidence)),
                        source_videos=list(set(existing.source_videos + mech.source_videos))
                    )
        return list(merged.values())

    def _merge_biomarker_effects(self, effect_lists: List[List[BiomarkerInsight]]) -> List[BiomarkerInsight]:
        """Merge biomarker effects while preserving unique information."""
        merged = {}
        for effects in effect_lists:
            for effect in effects:
                key = (effect.biomarker, effect.direction)
                if key not in merged:
                    merged[key] = effect
                else:
                    # Update existing effect with new information
                    existing = merged[key]
                    merged[key] = BiomarkerInsight(
                        biomarker=existing.biomarker,
                        direction=existing.direction,
                        magnitude=max(existing.magnitude, effect.magnitude),
                        confidence=max(existing.confidence, effect.confidence),
                        supporting_evidence=list(set(existing.supporting_evidence + effect.supporting_evidence)),
                        context=existing.context,  # Keep original context
                        age_related_change=existing.age_related_change
                    )
        return list(merged.values())

    def _calculate_merged_confidence(self, main_insight: DietaryInsight, related: List[DietaryInsight]) -> float:
        """Calculate confidence score for merged insights."""
        # Weight confidences by number of supporting studies
        confidences = [main_insight.confidence_score] + [r.confidence_score for r in related]
        study_counts = [len(main_insight.referenced_studies)] + [len(r.referenced_studies) for r in related]
        
        if not any(study_counts):  # No studies
            return sum(confidences) / len(confidences)
        
        # Weight by study count
        weighted_conf = sum(c * (s + 1) for c, s in zip(confidences, study_counts))
        total_weight = sum(s + 1 for s in study_counts)
        
        return min(weighted_conf / total_weight, 1.0)

    def _save_insights(self, insights: List[DietaryInsight], output_file: str):
        """Save insights to JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([self._insight_to_dict(i) for i in insights], f, indent=2)

    def _insight_to_dict(self, insight: DietaryInsight) -> dict:
        """Convert insight to dictionary for JSON serialization."""
        return {
            'key_finding': insight.key_finding,
            'mechanisms': [vars(m) for m in insight.mechanisms],
            'biomarker_effects': [vars(b) for b in insight.biomarker_effects],
            'studies': [vars(s) for s in insight.referenced_studies],
            'contradictions': insight.contradictions,
            'limitations': insight.limitations,
            'implications': insight.practical_implications,
            'confidence': insight.confidence_score,
            'sources': insight.source_videos,
            'category': insight.category,
            'interconnections': insight.interconnections
        }

    def _generate_comprehensive_report(self, insights: List[DietaryInsight], output_file: str):
        """Generate detailed markdown report of insights."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 🧬 Dietary Insights Analysis Report\n\n")
            
            # Write executive summary
            f.write("## 📊 Executive Summary\n\n")
            f.write(f"- Total unique insights analyzed: {len(insights)}\n")
            f.write(f"- Categories covered: {len(set(i.category for i in insights))}\n")
            f.write(f"- Total source videos: {len(set(sum([i.source_videos for i in insights], [])))}\n")
            f.write(f"- Average confidence score: {sum(i.confidence_score for i in insights)/len(insights):.2f}\n\n")
            
            # Key findings by category
            f.write("## 🎯 Key Findings by Category\n\n")
            by_category = defaultdict(list)
            for insight in insights:
                by_category[insight.category].append(insight)
            
            for category, category_insights in sorted(by_category.items()):
                f.write(f"### {category.title()}\n\n")
                
                # Sort by confidence
                for insight in sorted(category_insights, key=lambda x: x.confidence_score, reverse=True):
                    f.write(f"#### {insight.key_finding}\n")
                    f.write(f"**Confidence Score:** {'★' * int(insight.confidence_score * 5)}\n\n")
                    
                    if insight.mechanisms:
                        f.write("**Mechanisms:**\n")
                        for mechanism in insight.mechanisms:
                            f.write(f"- {mechanism.pathway}\n")
                            if mechanism.inputs:
                                f.write(f"  - Inputs: {', '.join(mechanism.inputs)}\n")
                            if mechanism.outputs:
                                f.write(f"  - Outputs: {', '.join(mechanism.outputs)}\n")
                        f.write("\n")
                    
                    if insight.biomarker_effects:
                        f.write("**Biomarker Effects:**\n")
                        for effect in insight.biomarker_effects:
                            f.write(f"- {effect.biomarker} {effect.direction} ")
                            if effect.magnitude != 0.3:  # Not default
                                f.write(f"by {effect.magnitude*100:.0f}% ")
                            f.write(f"(Age-related change: {effect.age_related_change})\n")
                        f.write("\n")
                    
                    if insight.referenced_studies:
                        f.write("**Supporting Studies:**\n")
                        for study in sorted(insight.referenced_studies, key=lambda x: x.confidence, reverse=True):
                            f.write(f"- {study.description}")
                            if study.population_size:
                                f.write(f" (n={study.population_size})")
                            f.write(f"\n  - Finding: {study.finding}\n")
                        f.write("\n")
                    
                    if insight.contradictions:
                        f.write("**Contradictory Findings:**\n")
                        for contra in insight.contradictions:
                            f.write(f"- {contra}\n")
                        f.write("\n")
                    
                    if insight.limitations:
                        f.write("**Limitations:**\n")
                        for limit in insight.limitations:
                            f.write(f"- {limit}\n")
                        f.write("\n")
                    
                    if insight.practical_implications:
                        f.write("**Practical Implications:**\n")
                        for impl in insight.practical_implications:
                            f.write(f"- {impl}\n")
                        f.write("\n")
                    
                    if insight.interconnections:
                        f.write("**Related Insights:**\n")
                        for conn in insight.interconnections:
                            f.write(f"- {conn}\n")
                        f.write("\n")
                    
                    f.write("---\n\n")

    def _generate_network_graph(self, insights: List[DietaryInsight], output_file: str):
        """Generate network graph of insights and their relationships."""
        nodes = []
        edges = []
        
        # Create nodes for insights
        for i, insight in enumerate(insights):
            nodes.append({
                'id': f'insight_{i}',
                'type': 'insight',
                'label': insight.key_finding[:50] + '...',
                'category': insight.category,
                'confidence': insight.confidence_score
            })
            
            # Create nodes for mechanisms and biomarkers
            for m, mechanism in enumerate(insight.mechanisms):
                mech_id = f'mechanism_{i}_{m}'
                nodes.append({
                    'id': mech_id,
                    'type': 'mechanism',
                    'label': mechanism.pathway[:50] + '...',
                    'confidence': mechanism.confidence
                })
                edges.append({
                    'source': f'insight_{i}',
                    'target': mech_id,
                    'type': 'has_mechanism'
                })
            
            for b, effect in enumerate(insight.biomarker_effects):
                bio_id = f'biomarker_{effect.biomarker}'
                if not any(n['id'] == bio_id for n in nodes):
                    nodes.append({
                        'id': bio_id,
                        'type': 'biomarker',
                        'label': effect.biomarker,
                        'age_change': effect.age_related_change
                    })
                edges.append({
                    'source': f'insight_{i}',
                    'target': bio_id,
                    'type': effect.direction
                })
        
        # Add edges for interconnections
        for i, insight in enumerate(insights):
            for conn in insight.interconnections:
                # Find related insight
                for j, other in enumerate(insights):
                    if other.key_finding in conn:
                        edges.append({
                            'source': f'insight_{i}',
                            'target': f'insight_{j}',
                            'type': 'related',
                            'description': conn
                        })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'nodes': nodes,
                'edges': edges
            }, f, indent=2)

    def _generate_evidence_matrix(self, insights: List[DietaryInsight], output_file: str):
        """Generate evidence matrix showing support for different findings."""
        # Collect all unique biomarkers and dietary factors
        biomarkers = set()
        dietary_factors = set()
        
        for insight in insights:
            for effect in insight.biomarker_effects:
                biomarkers.add(effect.biomarker)
            for mech in insight.mechanisms:
                dietary_factors.update(mech.inputs)
        
        # Create matrix
        matrix = {
            'biomarkers': sorted(biomarkers),
            'dietary_factors': sorted(dietary_factors),
            'relationships': [],
            'meta': {
                'generated_at': datetime.now().isoformat(),
                'total_insights': len(insights),
                'total_studies': sum(len(i.referenced_studies) for i in insights)
            }
        }
        
        # Fill relationships
        for biomarker in biomarkers:
            for factor in dietary_factors:
                supporting_insights = []
                
                for insight in insights:
                    # Check if insight connects this biomarker and factor
                    biomarker_effects = [e for e in insight.biomarker_effects if e.biomarker == biomarker]
                    mechanisms = [m for m in insight.mechanisms if factor in m.inputs]
                    
                    if biomarker_effects and mechanisms:
                        supporting_insights.append({
                            'finding': insight.key_finding,
                            'confidence': insight.confidence_score,
                            'evidence': [s.description for s in insight.referenced_studies]
                        })
                
                if supporting_insights:
                    matrix['relationships'].append({
                        'biomarker': biomarker,
                        'dietary_factor': factor,
                        'supporting_insights': supporting_insights,
                        'overall_confidence': sum(i['confidence'] for i in supporting_insights) / len(supporting_insights)
                    })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(matrix, f, indent=2)

def main():
    """Main function to run the insight extraction process."""
    print("Starting Enhanced Diet Insights Extraction")
    
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/version")
        if response.status_code != 200:
            raise Exception("Ollama is not responding properly")
    except Exception as e:
        print("\nError: Ollama is not running. Please start Ollama first.")
        print("\nInstallation instructions:")
        print("1. macOS: Download from https://ollama.ai/download/mac")
        print("   Linux: curl -fsSL https://ollama.ai/install.sh | sh")
        print("2. Start Ollama")
        print("3. Run: ollama pull mistral")
        return
    
    print("\nOllama is running and ready")
    
    # Initialize processor
    processor = ParallelInsightProcessor()
    
    try:
        processor.process_directory(
            "transcripts",
            "diet_insights.json"
        )
        
        print("\n✅ Processing completed successfully!")
        print("\nOutput files generated:")
        print("1. diet_insights.json - Raw data with all insights")
        print("2. diet_insights_report.md - Human-readable comprehensive report")
        print("3. diet_insights_network.json - Network visualization data")
        print("4. diet_insights_evidence.json - Evidence matrix")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        print("Please check the error message above and try again.")

if __name__ == "__main__":
    main()