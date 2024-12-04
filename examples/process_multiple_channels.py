"""Example script demonstrating processing multiple YouTube channels in parallel."""

from concurrent.futures import ThreadPoolExecutor
from typing import List

from youtube_transcript_retriever.core.channel import ChannelProcessor
from youtube_transcript_retriever.utils.logger import setup_logger

def process_channel(url: str) -> None:
    """Process a single channel."""
    processor = ChannelProcessor(
        channel_url=url,
        output_dir="transcripts",
        workers=4
    )
    processor.process_channel()

def main():
    # Setup logging
    logger = setup_logger(verbose=True)
    
    # List of channels to process
    channels = [
        "https://www.youtube.com/@veritasium",
        "https://www.youtube.com/@minutephysics",
        "https://www.youtube.com/@3blue1brown"
    ]
    
    # Process channels in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(channels)) as executor:
        executor.map(process_channel, channels)

if __name__ == "__main__":
    main()