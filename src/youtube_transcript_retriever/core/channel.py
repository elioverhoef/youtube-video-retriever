"""Channel processor module for handling YouTube channel transcript retrieval."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import re
import scrapetube
import requests

from .video import VideoProcessor
from ..utils.logger import get_logger
from ..utils.file_handler import ensure_directory
from ..exceptions import ChannelNotFoundError, InvalidChannelURLError

@dataclass
class ChannelProcessor:
    """Process YouTube channel to retrieve video transcripts."""
    channel_url: str
    output_dir: str = 'transcripts'
    workers: int = 4
    
    def __post_init__(self):
        self.logger = get_logger(__name__)
        self.channel_id = self._extract_channel_id(self.channel_url)
        self.channel_dir = Path(self.output_dir) / self._sanitize_channel_name()
        ensure_directory(self.channel_dir)
    
    def _extract_channel_id(self, url: str) -> str:
        """Extract channel ID from URL or return as is if it's an ID."""
        # Common URL patterns
        patterns = [
            r'youtube\.com/channel/([^/]+)',  # Regular channel URL
            r'youtube\.com/c/([^/]+)',       # Custom channel URL
            r'youtube\.com/@([^/]+)',       # Handle URL
            r'youtube\.com/user/([^/]+)'    # Legacy username URL
        ]
        
        for pattern in patterns:
            if match := re.search(pattern, url):
                return match.group(1)
        
        # Check if it's a valid channel ID format
        if re.match(r'^UC[\w-]{22}$', url):
            return url
            
        raise InvalidChannelURLError(
            f'Invalid channel URL or ID: {url}. '
            'Please provide a valid YouTube channel URL or ID.'
        )
    
    def _sanitize_channel_name(self) -> str:
        """Get channel name from ID and sanitize for filesystem."""
        try:
            # Use noembed to get channel info without API key
            response = requests.get(
                f'https://noembed.com/embed?url=https://youtube.com/channel/{self.channel_id}',
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if 'error' in data:
                raise ChannelNotFoundError(
                    f'Channel not found with ID: {self.channel_id}'
                )
                
            channel_name = data.get('author_name')
            if not channel_name:
                raise ValueError('Channel name not found in response')
                
        except requests.RequestException as e:
            self.logger.warning(f'Network error getting channel info: {e}')
            channel_name = self.channel_id
        except Exception as e:
            self.logger.warning(f'Error getting channel name: {e}')
            channel_name = self.channel_id
        
        # Sanitize name for filesystem
        sanitized = re.sub(r'[<>:"/\\|?*]', '', channel_name)
        sanitized = sanitized.replace(' ', '_')
        return sanitized[:255]  # Max filename length
    
    def _get_video_ids(self) -> List[str]:
        """Retrieve all video IDs from the channel."""
        self.logger.info(f'Fetching video IDs from channel {self.channel_id}...')
        try:
            videos = list(scrapetube.get_channel(self.channel_id))
            if not videos:
                raise ChannelNotFoundError(
                    f'No videos found for channel: {self.channel_id}'
                )
            return [video['videoId'] for video in videos]
        except Exception as e:
            raise ChannelNotFoundError(
                f'Error fetching videos from channel {self.channel_id}: {str(e)}'
            )
    
    def _process_video(self, video_id: str) -> Optional[Path]:
        """Process a single video with error handling."""
        try:
            processor = VideoProcessor(video_id, self.channel_dir)
            return processor.process()
        except Exception as e:
            self.logger.error(f'Error processing video {video_id}: {e}')
            return None
    
    def process_channel(self) -> None:
        """Process all videos in the channel using parallel workers."""
        try:
            video_ids = self._get_video_ids()
            self.logger.info(f'Found {len(video_ids)} videos')
            
            processed_count = 0
            error_count = 0
            
            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                # Submit all video processing tasks
                future_to_id = {
                    executor.submit(self._process_video, video_id): video_id
                    for video_id in video_ids
                }
                
                # Process results as they complete
                for future in as_completed(future_to_id):
                    video_id = future_to_id[future]
                    try:
                        result = future.result()
                        if result:
                            processed_count += 1
                            self.logger.info(
                                f'Processed video ({processed_count}/{len(video_ids)}): '
                                f'{result.name}'
                            )
                        else:
                            error_count += 1
                    except Exception as e:
                        error_count += 1
                        self.logger.error(f'Error processing video {video_id}: {e}')
            
            # Log final statistics
            self.logger.info('='*50)
            self.logger.info('Processing completed!')
            self.logger.info(f'Successfully processed: {processed_count} videos')
            self.logger.info(f'Failed to process: {error_count} videos')
            self.logger.info('='*50)
            
        except ChannelNotFoundError as e:
            self.logger.error(f'Channel error: {e}')
            raise
        except Exception as e:
            self.logger.error(f'Unexpected error: {e}')
            raise
