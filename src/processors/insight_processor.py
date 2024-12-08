from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import DBSCAN
from datetime import datetime

@dataclass
class Insight:
    type: str  # Finding/Protocol/Marker/Study
    content: str  # Main content
    metadata: Dict[str, str]  # Context, Effects, Limitations, etc.
    confidence: int  # Number of stars
    sources: List[str]  # Source attributions
    section: str  # Which section it belongs to
    tags: List[str]  # Extracted tags

class InsightParser:
    def __init__(self):
        # More flexible patterns that handle variable spacing and formats
        self.insight_pattern = r"(?:^|\n)\s*-\s*\*\*([^*]+)\*\*:\s*(.*?)(?=(?:\n\s*-\s*\*\*[^*]+\*\*:|\Z))"
        self.metadata_pattern = r"(?:^|\n)\s*-\s*([\w\s]+):\s*(.*?)(?=(?:\n\s*-|$))"
        self.tag_pattern = r"#(\w+(?:-\w+)*)"
        
    def parse_report(self, report_path: str) -> List[Insight]:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        insights = []
        
        # Process all insights
        for match in re.finditer(self.insight_pattern, content, re.DOTALL | re.MULTILINE):
            try:
                insight_type = match.group(1).strip()
                insight_text = match.group(2).strip()
                
                # Extract metadata from the insight text
                metadata = {}
                confidence = 0
                tags = []
                
                # Process metadata sections
                for meta_match in re.finditer(self.metadata_pattern, insight_text, re.MULTILINE):
                    key = meta_match.group(1).strip()
                    value = meta_match.group(2).strip()
                    
                    if key == 'Confidence':
                        confidence = value.count('⭐')
                    elif key == 'Tags':
                        tags = [tag.strip() for tag in re.findall(self.tag_pattern, value)]
                    else:
                        metadata[key] = value
                
                # Create insight object
                insight = Insight(
                    type='Marker' if insight_type == 'Marker' else 'Finding',
                    content=insight_text.split('\n')[0].strip(),  # First line is the main content
                    metadata=metadata,
                    confidence=confidence,
                    sources=[],
                    section=self._determine_section(insight_type, tags, metadata),
                    tags=tags
                )
                
                insights.append(insight)
                
            except Exception as e:
                print(f"Error parsing insight: {e}")
                continue
                
        return insights
        
    def _determine_section(self, insight_type: str, tags: List[str], metadata: Dict[str, str]) -> str:
        """Determine the section based on insight type, tags, and metadata"""
        if insight_type == 'Marker':
            return 'Health Markers'
        
        # Check tags and metadata for section hints
        all_text = ' '.join([insight_type] + tags + list(metadata.values())).lower()
        
        if any(word in all_text for word in ['diet', 'food', 'nutrition', 'eating']):
            return 'Diet Insights'
        elif any(word in all_text for word in ['supplement', 'vitamin', 'mineral']):
            return 'Supplements'
        elif any(word in all_text for word in ['study', 'research', 'trial']):
            return 'Scientific Methods'
            
        # Default to Health Markers if no other match
        return 'Health Markers'

class SimilarityDetector:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def find_similar_insights(self, insights: List[Insight]) -> List[List[Insight]]:
        if not insights:
            return []
            
        # Create embeddings for each insight
        texts = [f"{insight.content} {' '.join(insight.metadata.values())}" for insight in insights]
        embeddings = self.model.encode(texts)
        
        # Cluster similar insights
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
        labels = clustering.fit_predict(embeddings)
        
        # Group insights by cluster
        clusters = {}
        for label, insight in zip(labels, insights):
            if label != -1:  # -1 represents noise in DBSCAN
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(insight)
                
        return list(clusters.values())

class InsightMerger:
    def merge_cluster(self, similar_insights: List[Insight]) -> Insight:
        if len(similar_insights) == 1:
            return similar_insights[0]
            
        # Take the insight with highest confidence as base
        base_insight = max(similar_insights, key=lambda x: x.confidence)
        
        # Combine all sources and tags
        all_sources = set()
        all_tags = set()
        for insight in similar_insights:
            all_sources.update(insight.sources)
            all_tags.update(insight.tags)
            
        # Merge metadata
        merged_metadata = {}
        for key in base_insight.metadata.keys():
            values = []
            for insight in similar_insights:
                if key in insight.metadata:
                    values.append(insight.metadata[key])
            merged_metadata[key] = '; '.join(set(values))
            
        return Insight(
            type=base_insight.type,
            content=base_insight.content,
            metadata=merged_metadata,
            confidence=base_insight.confidence,
            sources=list(all_sources),
            section=base_insight.section,
            tags=list(all_tags)
        )

class ReportProcessor:
    def __init__(self):
        self.parser = InsightParser()
        self.detector = SimilarityDetector()
        self.merger = InsightMerger()
        
    def process_report(self, input_path: str, output_path: str):
        # Parse insights
        insights = self.parser.parse_report(input_path)
        
        # Process each section separately
        processed_insights = []
        for section in set(insight.section for insight in insights):
            section_insights = [i for i in insights if i.section == section]
            
            # Find similar insights
            clusters = self.detector.find_similar_insights(section_insights)
            
            # Merge similar insights
            for cluster in clusters:
                merged = self.merger.merge_cluster(cluster)
                processed_insights.append(merged)
                
            # Add non-clustered insights
            clustered_ids = {id(insight) for cluster in clusters for insight in cluster}
            non_clustered = [i for i in section_insights if id(i) not in clustered_ids]
            processed_insights.extend(non_clustered)
            
        # Generate new report
        self._generate_report(processed_insights, output_path, input_path)
        
    def _generate_report(self, insights: List[Insight], output_path: str, input_path: str):
        # Read original report
        with open(input_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Split content at key sections
        header_content = original_content.split('## Diet Insights')[0]
        sources_section = original_content[original_content.find('## Sources'):]
        
        # Start with original header
        report = [header_content.strip()]
        
        # Add main sections with processed insights
        main_sections = ["Diet Insights", "Supplements", "Scientific Methods", "Health Markers"]
        for section in main_sections:
            report.append(f"\n## {section}\n")
            section_insights = [i for i in insights if i.section == section]
            section_insights.sort(key=lambda x: x.confidence, reverse=True)
            
            for insight in section_insights:
                report.append(f"- **{insight.type}**: {insight.content}")
                for key, value in insight.metadata.items():
                    report.append(f"    - {key}: {value}")
                if insight.sources:
                    report.append(f"    _(Sources: {'; '.join(insight.sources)})_")
                report.append("")
        
        # Add original Sources section
        report.append(sources_section)
        
        # Write report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
            
    def _generate_source_summary(self, insights: List[Insight]) -> str:
        """Generate a summary for insights from a single source."""
        # Combine all insight content and metadata
        all_text = " ".join([
            f"{insight.content} {' '.join(insight.metadata.values())}"
            for insight in insights
        ])
        # Return first few sentences as summary
        sentences = all_text.split('. ')
        return ". ".join(sentences[:3]) + "."
