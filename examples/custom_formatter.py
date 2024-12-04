"""Example demonstrating how to customize transcript formatting."""

from pathlib import Path
from typing import Dict, List, Any

from youtube_transcript_retriever.core.channel import ChannelProcessor
from youtube_transcript_retriever.utils.logger import setup_logger

def custom_format_transcript(transcript_data: List[Dict[str, Any]]) -> str:
    """Custom transcript formatter that includes timestamps for every line."""
    formatted_text = ""
    
    for item in transcript_data:
        # Format timestamp as [MM:SS]
        seconds = int(item['start'])
        minutes = seconds // 60
        seconds = seconds % 60
        timestamp = f"[{minutes:02d}:{seconds:02d}]"
        
        # Add formatted line
        formatted_text += f"{timestamp} {item['text']}\n"
    
    return formatted_text

def custom_create_markdown(video_info: Dict[str, str], transcript_text: str) -> str:
    """Custom markdown formatter with additional metadata."""
    return f"""# {video_info['title']}

## Video Information
- **Channel:** {video_info['author_name']}
- **Published:** {video_info.get('upload_date', 'Unknown')}
- **Link:** {video_info['url']}

## Transcript
_Each line includes a timestamp [MM:SS]_

{transcript_text}

---
_Transcript retrieved using YouTube Transcript Retriever_
"""

def main():
    # Setup logging
    logger = setup_logger(verbose=True)
    
    # Create processor with custom formatters
    processor = ChannelProcessor(
        channel_url="https://www.youtube.com/@veritasium",
        output_dir="transcripts",
        workers=4
    )
    
    # Override default formatters
    processor._format_transcript = custom_format_transcript
    processor._create_markdown = custom_create_markdown
    
    # Process channel
    processor.process_channel()

if __name__ == "__main__":
    main()