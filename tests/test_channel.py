"""Tests for channel processing functionality."""

import pytest
from pathlib import Path

from youtube_transcript_retriever.core.channel import ChannelProcessor
from youtube_transcript_retriever.exceptions import InvalidChannelURLError

def test_channel_url_extraction():
    """Test extracting channel IDs from various URL formats."""
    test_cases = [
        (
            "https://www.youtube.com/channel/UC123",
            "UC123"
        ),
        (
            "https://www.youtube.com/c/Veritasium",
            "Veritasium"
        ),
        (
            "https://www.youtube.com/@veritasium",
            "veritasium"
        ),
        (
            "UC123",  # Direct channel ID
            "UC123"
        ),
    ]
    
    for url, expected in test_cases:
        processor = ChannelProcessor(url)
        assert processor.channel_id == expected

def test_invalid_channel_url():
    """Test handling of invalid channel URLs."""
    invalid_urls = [
        "https://youtube.com/invalid",
        "https://youtube.com/watch?v=123",
        "not_a_url",
    ]
    
    for url in invalid_urls:
        with pytest.raises(InvalidChannelURLError):
            ChannelProcessor(url)

def test_output_directory_creation(tmp_path):
    """Test creation of output directory structure."""
    processor = ChannelProcessor(
        "UC123",
        output_dir=str(tmp_path)
    )
    
    # Channel directory should be created
    assert processor.channel_dir.exists()
    assert processor.channel_dir.is_dir()