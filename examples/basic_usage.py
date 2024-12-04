"""Example script demonstrating basic usage of the YouTube Transcript Retriever."""

from youtube_transcript_retriever.core.channel import ChannelProcessor
from youtube_transcript_retriever.utils.logger import setup_logger

def main():
    # Setup logging
    logger = setup_logger(verbose=True)
    
    # Example channel URL (Veritasium)
    channel_url = "https://www.youtube.com/c/veritasium"
    
    # Create processor instance
    processor = ChannelProcessor(
        channel_url=channel_url,
        output_dir="transcripts",
        workers=4  # Use 4 parallel workers
    )
    
    # Process all videos in the channel
    processor.process_channel()

if __name__ == "__main__":
    main()