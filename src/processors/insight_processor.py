from dataclasses import dataclass
from typing import List, Dict, Optional
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import json
import logging
from ..models.gemini_client import GeminiClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


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
        # Capture both type and content
        self.insight_pattern = r"(?:^|\n)\s*-\s*\*\*(Finding|Protocol|Marker|Study|Study Type)\*\*:\s*(.*?)(?=\n\s*-\s*\*\*(?:Finding|Protocol|Marker|Study|Study Type)\*\*:|\Z)"

    def parse_report(self, report_path: str) -> List[Insight]:
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()

        insights = []

        # Process insights with modified pattern that captures type
        for match in re.finditer(self.insight_pattern, content, re.DOTALL):
            insight_type, insight_text = match.groups()
            insight = self._parse_insight(insight_type, insight_text)
            if insight:
                insights.append(insight)

        return insights

    def _parse_insight(self, insight_type: str, text: str) -> Optional[Insight]:
        # Clean up the text and split into lines
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            return None

        # First line is the main content
        main_content = lines[0].strip()

        # Process metadata lines (they start with '-' or have ':')
        metadata = {}
        for line in lines[1:]:
            line = line.strip("- ").strip()
            if ":" in line:
                key, value = [x.strip() for x in line.split(":", 1)]
                metadata[key] = value

        # Extract confidence
        confidence = 0
        confidence_text = metadata.get("Confidence", "")
        if confidence_text:
            confidence = confidence_text.count("⭐")

        # Extract tags
        tags = []
        tags_text = metadata.get("Tags", "")
        if tags_text:
            tags = [tag.strip() for tag in tags_text.split("#") if tag.strip()]

        # Map type to section directly
        section_mapping = {
            "Finding": "Diet Insights",
            "Protocol": "Supplements",
            "Study": "Scientific Methods",
            "Study Type": "Scientific Methods",
            "Marker": "Health Markers",
        }

        # Clean up insight type
        if insight_type == "Study Type":
            insight_type = "Study"

        return Insight(
            type=insight_type,
            content=main_content,
            metadata=metadata,
            confidence=confidence,
            sources=[],
            section=section_mapping.get(insight_type, "Other"),
            tags=tags,
        )


class SimilarityDetector:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def find_similar_insights(self, insights: List[Insight]) -> List[List[Insight]]:
        if not insights:
            return []

        # Create embeddings for each insight
        texts = [
            f"{insight.content} {' '.join(insight.metadata.values())}"
            for insight in insights
        ]
        embeddings = self.model.encode(texts)

        clustering = DBSCAN(eps=0.35, min_samples=2, metric="cosine")
        labels = clustering.fit_predict(embeddings)

        # Group insights by cluster
        clusters = {}
        for label, insight in zip(labels, insights):
            if label != -1:  # -1 represents noise in DBSCAN
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(insight)
        # Print number of clusters found
        print(f"Found {len(clusters)} clusters containing {sum(len(cluster) for cluster in clusters.values())} insights")
        return list(clusters.values())


class InsightMerger:
    def __init__(self, gemini_client: GeminiClient):
        self.gemini_client = gemini_client
        self.merge_prompt = """
Given these similar insights, merge them into a single comprehensive insight that preserves ALL information while avoiding redundancy.
Combine semantically identical information but ensure no unique details are lost.

IMPORTANT: The output must be valid JSON with NO trailing commas.

Input Insights:
{insights_json}

Output a single merged insight in this exact JSON format:
{{
    "type": "string",
    "content": "string",
    "metadata": {{
        "Context": "string",
        "Effects": "string",
        "Limitations": "string",
        "Confidence": "string",
        "Tags": "string"
        // preserve any other metadata fields
    }},
    "confidence": number,
    "sources": ["string"],
    "section": "string",
    "tags": ["string"]
}}
"""

    def merge_cluster(self, similar_insights: List[Insight]) -> Insight:
        if len(similar_insights) == 1:
            return similar_insights[0]

        # Convert insights to JSON-friendly format for the prompt
        insights_json = []
        for insight in similar_insights:
            insights_json.append(
                {
                    "type": insight.type,
                    "content": insight.content,
                    "metadata": insight.metadata,
                    "confidence": insight.confidence,
                    "sources": insight.sources,
                    "section": insight.section,
                    "tags": insight.tags,
                }
            )

        # Generate prompt with insights
        prompt = self.merge_prompt.format(
            insights_json=json.dumps(insights_json, indent=2)
        )

        try:
            # Get merged insight from LLM
            merged_json_str, _ = self.gemini_client.get_completion(prompt)
            merged_data = json.loads(
                merged_json_str.replace("```json", "").replace("```", "")
            )
            print(merged_data)

            # Convert back to Insight object
            return Insight(
                type=merged_data["type"],
                content=merged_data["content"],
                metadata=merged_data["metadata"],
                confidence=merged_data["confidence"],
                sources=merged_data["sources"],
                section=merged_data["section"],
                tags=merged_data["tags"],
            )
        except Exception as e:
            logging.error(f"LLM merging failed: {str(e)}")
            # Fallback to original merging logic if LLM fails
            return self._fallback_merge(similar_insights)

    def _fallback_merge(self, similar_insights: List[Insight]) -> Insight:
        # Original merging logic as fallback
        base_insight = max(similar_insights, key=lambda x: x.confidence)
        all_sources = set()
        all_tags = set()
        for insight in similar_insights:
            all_sources.update(insight.sources)
            all_tags.update(insight.tags)

        merged_metadata = {}
        for key in base_insight.metadata.keys():
            values = []
            for insight in similar_insights:
                if key in insight.metadata:
                    values.append(insight.metadata[key])
            merged_metadata[key] = "; ".join(set(values))

        return Insight(
            type=base_insight.type,
            content=base_insight.content,
            metadata=merged_metadata,
            confidence=base_insight.confidence,
            sources=list(all_sources),
            section=base_insight.section,
            tags=list(all_tags),
        )


class ReportProcessor:
    def __init__(self, gemini_client: GeminiClient):
        self.parser = InsightParser()
        self.detector = SimilarityDetector()
        self.merger = InsightMerger(gemini_client)
        self.executor = ThreadPoolExecutor(max_workers=8)

    def process_report(self, input_path: str, output_path: str):
        # Parse insights
        insights = self.parser.parse_report(input_path)

        # Process each section separately
        processed_insights = []
        futures = []

        for section in set(insight.section for insight in insights):
            section_insights = [i for i in insights if i.section == section]

            # Find similar insights
            clusters = self.detector.find_similar_insights(section_insights)

            # Submit merge tasks to thread pool
            for cluster in clusters:
                if len(cluster) > 1:  # Only submit if actually needs merging
                    future = self.executor.submit(self.merger.merge_cluster, cluster)
                    futures.append(future)
                else:
                    processed_insights.append(cluster[0])

            # Add non-clustered insights
            clustered_ids = {id(insight) for cluster in clusters for insight in cluster}
            non_clustered = [i for i in section_insights if id(i) not in clustered_ids]
            processed_insights.extend(non_clustered)

            # Collect results from futures with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Merging insights"):
                try:
                    merged_insight = future.result()
                    processed_insights.append(merged_insight)
                except Exception as e:
                    logging.error(f"Error in merge task: {str(e)}")

        # Generate new report
        self._generate_report(processed_insights, output_path, input_path)

    def __del__(self):
        self.executor.shutdown(wait=True)

    def _generate_report(
        self, insights: List[Insight], output_path: str, input_path: str
    ):
        # Read original report and extract only the content we want
        with open(input_path, "r", encoding="utf-8") as f:
            original_content = f.read()

        # Extract only the content between Diet Insights and Sources
        if "## Diet Insights" in original_content:
            content = original_content[original_content.find("## Diet Insights") :]
            if "## Sources" in content:
                content = content[: content.find("## Sources")]

        # Start fresh report
        report = []

        # Add main sections with processed insights
        main_sections = [
            "Diet Insights",
            "Supplements",
            "Health Markers",
        ]
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

        # Write report
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))

    def _generate_source_summary(self, insights: List[Insight]) -> str:
        """Generate a summary for insights from a single source."""
        # Combine all insight content and metadata
        all_text = " ".join(
            [
                f"{insight.content} {' '.join(insight.metadata.values())}"
                for insight in insights
            ]
        )
        # Return first few sentences as summary
        sentences = all_text.split(". ")
        return ". ".join(sentences[:3]) + "."
