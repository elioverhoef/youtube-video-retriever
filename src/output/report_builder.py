from pathlib import Path
import yaml
import logging
from typing import List, Dict
from datetime import datetime
import re
from difflib import SequenceMatcher

class ReportBuilder:
    def __init__(self):
        # Load config
        config_path = Path("config/config.yaml")
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self.logger = logging.getLogger(__name__)
        self.similarity_threshold = 0.8  # Configurable similarity threshold
        
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        return SequenceMatcher(None, text1, text2).ratio()
        
    def _merge_similar_insights(self, insights: List[str]) -> List[str]:
        """Merge similar insights while preserving sources."""
        if not insights:
            return []
            
        merged = []
        used = set()
        
        for i, insight1 in enumerate(insights):
            if i in used:
                continue
                
            similar_insights = []
            for j, insight2 in enumerate(insights[i+1:], i+1):
                if j not in used and self._calculate_similarity(insight1, insight2) > self.similarity_threshold:
                    similar_insights.append(insight2)
                    used.add(j)
                    
            if similar_insights:
                sources = []
                for insight in [insight1] + similar_insights:
                    source_match = re.search(r'_\(Source: ([^)]+)\)_', insight)
                    if source_match:
                        sources.append(source_match.group(1))
                
                base_insight = max([insight1] + similar_insights, 
                                 key=lambda x: len(re.findall('⭐', x)))
                base_insight = re.sub(r'_\(Source: [^)]+\)_', 
                                    f'_(Sources: {"; ".join(sources)})_',
                                    base_insight)
                merged.append(base_insight)
            else:
                merged.append(insight1)
                
        return merged
        
    def _sort_by_confidence(self, insights: List[str]) -> List[str]:
        """Sort insights by confidence score (higher first)."""
        def get_confidence(insight: str) -> int:
            stars = re.findall('⭐', insight)
            return len(stars)
            
        return sorted(insights, key=get_confidence, reverse=True)
        
    def _build_section(self, section: str, results: List[Dict[str, str]]) -> List[str]:
        """Build a single section of the report with improved organization."""
        section_content = [f"\n## {section}\n"]
        
        # Special handling for Executive Summary and Quick Reference
        if section in ["Executive Summary", "Quick Reference"]:
            return self._build_summary_section(section, results)
            
        insights = []
        for result in results:
            content = result["content"]
            source = result["source"]
            
            # Extract the relevant section from the content
            section_pattern = f"## {section}\n"
            if section_pattern in content:
                start = content.index(section_pattern) + len(section_pattern)
                next_section = content.find("\n## ", start)
                
                if next_section != -1:
                    section_text = content[start:next_section].strip()
                else:
                    section_text = content[start:].strip()
                    
                if section_text:
                    # Process and format the content
                    formatted_text = self._format_content(section_text, source)
                    insights.extend(formatted_text.split("\n\n"))
                    
        # Merge similar insights and sort by confidence
        merged_insights = self._merge_similar_insights(insights)
        sorted_insights = self._sort_by_confidence(merged_insights)
        
        section_content.extend(sorted_insights)
        return section_content
        
    def build_report(self, results: List[Dict[str, str]]) -> str:
        """Build a comprehensive markdown report from all processed transcripts."""
        if not results:
            return "No insights found."
            
        # Start with report header
        report = [
            "# Health and Longevity Insights Report",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Table of Contents\n"
        ]
        
        # Add table of contents
        sections = self.config["output"]["report_sections"]
        for section in sections:
            report.append(f"- [{section}](#{section.lower().replace(' ', '-')})")
            
        report.append("\n---\n")
        
        # Add legend
        report.extend(self._build_legend())
        
        # Process each section
        for section in sections:
            report.extend(self._build_section(section, results))
            
        # Add sources section
        report.extend(self._build_sources_section(results))
        
        return "\n".join(report)
        
    def _build_legend(self) -> List[str]:
        """Build a legend explaining the confidence scores."""
        legend = [
            "## Legend\n",
            "### Confidence Scores",
            "Higher scores indicate stronger evidence:"
        ]
        
        # Add confidence symbols
        for score, stars in self.config["output"]["confidence_display"].items():
            legend.append(f"- {score}: {stars}")
            
        legend.append("\n---\n")
        return legend
        
    def _format_content(self, text: str, source: str) -> str:
        """Format content with source attribution."""
        lines = []
        current_item = []
        
        for line in text.split("\n"):
            # Main bullet points
            if line.startswith("- Finding:") or line.startswith("- Protocol:") or line.startswith("- Marker:"):
                if current_item:
                    lines.append("\n".join(current_item))
                    current_item = []
                current_item.append(line)
            # Sub-points should be indented
            elif line.startswith("- Context:") or line.startswith("- Effects:") or \
                 line.startswith("- Limitations:") or line.startswith("- Timeframe:"):
                current_item.append("    " + line)
            # Confidence line should be indented and include source
            elif line.startswith("- Confidence:"):
                current_item.append(f"    {line} _(Source: {source})_")
            else:
                # Other lines maintain their current indentation level
                if current_item and (line.startswith("    ") or not line.strip()):
                    current_item.append(line)
                else:
                    current_item.append("    " + line.lstrip())
                
        # Add the last item
        if current_item:
            lines.append("\n".join(current_item))
            
        return "\n\n".join(lines)
        
    def _build_summary_section(self, section: str, results: List[Dict[str, str]]) -> List[str]:
        """Build Executive Summary or Quick Reference section."""
        section_content = [f"\n## {section}\n"]
        
        for result in results:
            content = result["content"]
            source = result["source"]
            
            section_pattern = f"## {section}\n"
            if section_pattern in content:
                start = content.index(section_pattern) + len(section_pattern)
                next_section = content.find("\n## ", start)
                
                if next_section != -1:
                    section_text = content[start:next_section].strip()
                else:
                    section_text = content[start:].strip()
                    
                if section_text:
                    # Add source reference but keep the format simpler for summaries
                    section_content.append(f"{section_text} _(Source: {source})_\n")
                    
        return section_content
        
    def _build_sources_section(self, results: List[Dict[str, str]]) -> List[str]:
        """Build the sources section listing all processed transcripts."""
        sources = ["\n## Sources\n"]
        
        for result in results:
            source = result["source"]
            model = result["model"]
            sources.append(f"- {source} _(Processed with {model})_")
            
        return sources
        
    def save_report(self, report: str, output_path: str = "output/insights_report.md"):
        """Save the report to a file."""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report)
                
            self.logger.info(f"Report saved to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving report: {str(e)}") 