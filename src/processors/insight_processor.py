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
        # Match both top-level and nested findings
        self.insight_pattern = r"(?:^|\n)- \*\*(?:Finding|Protocol|Marker|Study|Study Type)\*\*: (.*?)(?=(?:\n- \*\*(?:Finding|Protocol|Marker|Study|Study Type)\*\*:|\Z))"
        self.nested_finding_pattern = r"(?:^|\n)\s+\*\*(?:Finding|Protocol|Marker|Study|Study Type)\*\*: (.*?)(?=(?:\n\s+\*\*(?:Finding|Protocol|Marker|Study|Study Type)\*\*:|\n- |\Z))"
        
    def parse_report(self, report_path: str) -> List[Insight]:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        insights = []
        
        # Process top-level insights
        for match in re.finditer(self.insight_pattern, content, re.DOTALL):
            insight_text = match.group(1)
            # Also look for nested findings within this insight
            nested_matches = re.finditer(self.nested_finding_pattern, insight_text, re.DOTALL)
            
            # First process the main insight
            main_insight = self._parse_insight(insight_text)
            if main_insight:
                insights.append(main_insight)
            
            # Then process any nested findings
            for nested_match in nested_matches:
                nested_text = nested_match.group(1)
                nested_insight = self._parse_insight(nested_text)
                if nested_insight:
                    insights.append(nested_insight)
                    
        return insights
        
    def _parse_insight(self, text: str) -> Optional[Insight]:
        # Split into lines and clean up
        lines = []
        current_line = ""
        
        # Handle multi-line content more carefully
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                if current_line:
                    lines.append(current_line)
                    current_line = ""
            elif line.startswith('- '):
                if current_line:
                    lines.append(current_line)
                current_line = line[2:]  # Remove the "- " prefix
            else:
                if current_line:
                    current_line += " " + line
                else:
                    current_line = line
                    
        if current_line:
            lines.append(current_line)
            
        if not lines:
            return None
            
        # Extract main content
        main_content = lines[0].strip()
        
        # Extract metadata
        metadata = {}
        for line in lines[1:]:
            if ':' in line:
                key_value = line.split(':', 1)
                if len(key_value) == 2:
                    key, value = key_value
                    metadata[key.strip()] = value.strip()
                    
        # Extract confidence
        confidence = 0
        confidence_text = metadata.get('Confidence', '')
        if confidence_text:
            confidence = confidence_text.count('⭐')
            
        # Extract tags
        tags = []
        tags_text = metadata.get('Tags', '')
        if tags_text:
            tags = [tag.strip() for tag in tags_text.split('#') if tag.strip()]
            
        # Determine type and section
        type_and_section = self._determine_type_and_section(main_content, tags, metadata)
        
        return Insight(
            type=type_and_section['type'],
            content=main_content,
            metadata=metadata,
            confidence=confidence,
            sources=[],
            section=type_and_section['section'],
            tags=tags
        )
        
    def _determine_type_and_section(self, content: str, tags: List[str], metadata: Dict[str, str]) -> Dict[str, str]:
        # Keywords for classification
        diet_keywords = {
            'diet', 'calorie', 'food', 'nutrition', 'intake', 'eating', 'meal',
            'protein', 'carb', 'fat', 'fish', 'meat', 'vegetable', 'fruit',
            'dairy', 'egg', 'oil', 'nut', 'seed', 'grain', 'bread', 'sugar'
        }
        supplement_keywords = {
            'supplement', 'vitamin', 'mineral', 'compound', 'capsule', 'tablet',
            'dose', 'supplementation', 'nutrient', 'amino acid', 'enzyme'
        }
        marker_keywords = {
            'marker', 'measurement', 'level', 'test', 'score', 'blood',
            'biomarker', 'hormone', 'enzyme', 'cholesterol', 'glucose',
            'insulin', 'cortisol', 'testosterone', 'estrogen'
        }
        
        content_lower = content.lower()
        tags_lower = {tag.lower() for tag in tags}
        
        # Check context in metadata
        context = metadata.get('Context', '').lower()
        
        # Combine all text for keyword matching
        all_text = f"{content_lower} {context} {' '.join(tags_lower)}"
        
        # Determine type and section
        if any(keyword in all_text for keyword in diet_keywords):
            return {'type': 'Finding', 'section': 'Diet Insights'}
        elif any(keyword in all_text for keyword in supplement_keywords):
            return {'type': 'Protocol', 'section': 'Supplements'}
        elif any(keyword in all_text for keyword in marker_keywords):
            return {'type': 'Marker', 'section': 'Health Markers'}
        else:
            return {'type': 'Study', 'section': 'Scientific Methods'}

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
        self._generate_report(processed_insights, output_path)
        
    def _generate_report(self, insights: List[Insight], output_path: str):
        sections = ["Diet Insights", "Supplements", "Scientific Methods", "Health Markers"]
        
        report = ["# Processed Health and Longevity Insights Report\n"]
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Add table of contents
        report.append("## Table of Contents\n")
        for section in sections:
            report.append(f"- [{section}](#{section.lower().replace(' ', '-')})")
        report.append("\n---\n")
        
        # Add each section
        for section in sections:
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
                
        # Write report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
