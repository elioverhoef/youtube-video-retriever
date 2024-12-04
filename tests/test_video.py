"""Tests for video processing functionality."""

import pytest
from pathlib import Path

from youtube_transcript_retriever.core.video import VideoProcessor
from youtube_transcript_retriever.exceptions import VideoNotFoundError

def test_video_info_retrieval():
    """Test retrieving video information."""
    # Use a known video ID
    processor = VideoProcessor(
        "dQw4w9WgXcQ",  # Never Gonna Give You Up
        Path(".")
    )
    
    info = processor._get_video_info()
    assert info['title']
    assert info['author_name']
    assert 'url' in info

def test_invalid_video_id():
    """Test handling of invalid video IDs."""
    processor = VideoProcessor(
        "invalid_video_id",
        Path(".")
    )
    
    with pytest.raises(VideoNotFoundError):
        processor._get_video_info()

def test_transcript_formatting():
    """Test transcript text formatting."""
    from youtube_transcript_retriever.utils.formatter import format_transcript
    
    # Sample transcript data
    transcript_data = [
        {"text": "Hello", "start": 0.0},
        {"text": "world!", "start": 1.0},
        {"text": "This is a test.", "start": 2.0},
    ]
    
    formatted = format_transcript(transcript_data)
    assert formatted.strip()
    assert "." in formatted  # Should have proper punctuation
    assert formatted[0].isupper()  # Should be capitalized