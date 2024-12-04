"""Example showing how to filter which videos to process."""

from datetime import datetime
from typing import Optional
import requests

from youtube_transcript_retriever.core.channel import ChannelProcessor
from youtube_transcript_retriever.utils.logger import setup_logger

def get_video_date(video_id: str) -> Optional[datetime]:
    """Get upload date of a video."""
    try:
        response = requests.get(
            f'https://noembed.com/embed?url=https://www.youtube.com/watch?v={video_id}'
        )
        data = response.json()
        date_str = data.get('upload_date')
        if date_str:
            return datetime.strptime(date_str, '%Y-%m-%d')
    except Exception:
        return None
    return None

class FilteredChannelProcessor(ChannelProcessor):
    """Custom processor that only processes recent videos."""
    def _get_video_ids(self) -> List[str]:
        # Get all video IDs
        video_ids = super()._get_video_ids()
        
        # Filter to only include videos from 2024
        filtered_ids = []
        for vid_id in video_ids:
            if upload_date := get_video_date(vid_id):
                if upload_date.year >= 2024:
                    filtered_ids.append(vid_id)
        
        self.logger.info(
            f'Filtered {len(video_ids)} videos down to {len(filtered_ids)} '
            'videos from 2024'
        )
        return filtered_ids

def main():
    # Setup logging
    logger = setup_logger(verbose=True)
    
    # Create filtered processor
    processor = FilteredChannelProcessor(
        channel_url="https://www.youtube.com/@veritasium",
        output_dir="transcripts",
        workers=4
    )
    
    # Process only recent videos
    processor.process_channel()

if __name__ == "__main__":
    main()