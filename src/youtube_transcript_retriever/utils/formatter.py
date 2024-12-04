"""Text formatting utilities for transcript processing."""

import re
from typing import List, Dict, Any
from datetime import timedelta

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    return str(timedelta(seconds=int(seconds)))

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove speaker labels like "[Speaker 1:]" or "Speaker:" 
    text = re.sub(r'\[.*?\]|\b\w+:', '', text)
    
    # Remove common transcript artifacts
    artifacts = [
        r'\(.*?\)',          # Text in parentheses
        r'\[.*?\]',          # Text in square brackets
        r'♪.*?♪',            # Music notes and lyrics
        r'\[Music\]',        # Music indicators
        r'\[Applause\]',     # Sound effects
        r'\[Laughter\]',     # Audience reactions
    ]
    for artifact in artifacts:
        text = re.sub(artifact, '', text)
    
    return text.strip()

def combine_sequential_text(transcript_data: List[Dict[str, Any]]) -> str:
    """Combine sequential text entries into proper sentences."""
    full_text = ""
    buffer = ""
    current_speaker = None
    
    for item in transcript_data:
        text = clean_text(item['text'])
        if not text:  # Skip empty text after cleaning
            continue
            
        timestamp = format_timestamp(item['start'])
        
        # Handle potential speaker change (if detected)
        if ':' in text and len(text.split(':', 1)) == 2:
            potential_speaker, content = text.split(':', 1)
            if len(potential_speaker.split()) <= 3:  # Assume it's a speaker if ≤3 words
                if current_speaker != potential_speaker:
                    # New speaker - flush buffer and start new section
                    if buffer:
                        full_text += f"{buffer}\n\n"
                        buffer = ""
                    current_speaker = potential_speaker
                    buffer += f"[{timestamp}] **{current_speaker}:** {content.strip()} "
                    continue
        
        # Handle sentence endings
        if text.rstrip().endswith(('.', '!', '?')):
            buffer += f"{text} "
            if current_speaker:
                full_text += f"[{timestamp}] **{current_speaker}:** {buffer}\n\n"
            else:
                full_text += f"[{timestamp}] {buffer}\n\n"
            buffer = ""
        else:
            buffer += f"{text} "
    
    # Add any remaining text
    if buffer:
        if current_speaker:
            full_text += f"[{timestamp}] **{current_speaker}:** {buffer}\n\n"
        else:
            full_text += f"[{timestamp}] {buffer}\n\n"
    
    return full_text.strip()

def format_transcript(transcript_data: List[Dict[str, Any]]) -> str:
    """Format transcript data into clean, readable text."""
    # Combine text entries into proper sentences
    full_text = combine_sequential_text(transcript_data)
    
    # Post-processing cleanup
    # Remove multiple newlines
    full_text = re.sub(r'\n\s*\n', '\n\n', full_text)
    
    # Ensure proper capitalization
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    sentences = [s[0].upper() + s[1:] if s else s for s in sentences]
    full_text = ' '.join(sentences)
    
    return full_text.strip()

def create_markdown(video_info: Dict[str, str], transcript_text: str) -> str:
    """Create a formatted markdown document."""
    return f"""# {video_info['title']}

**Author:** {video_info['author_name']}
**Date:** {video_info.get('upload_date', 'Unknown')}
**Source:** [{video_info['url']}]({video_info['url']})

## Transcript

{transcript_text}

---
*This transcript was automatically generated and formatted from the YouTube video "{video_info['title']}".*

*Timestamps are included for reference. Speakers are marked in bold when detected.*
"""