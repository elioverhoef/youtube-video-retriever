import os
import json
import scrapetube
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from tqdm import tqdm
import re
import concurrent.futures
from pathlib import Path

# Constants
CHANNEL_ID = "UCT1UMLpZ_CrQ_8I431K0b-g"  # Michael Lustgarden's channel ID
TRANSCRIPTS_DIR = "transcripts"
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/videos"

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_video_info(video_id):
    """Get video title and other metadata from YouTube Data API"""
    try:
        # Note: In production, you should use an API key
        response = requests.get(f"https://noembed.com/embed?url=https://www.youtube.com/watch?v={video_id}")
        data = response.json()
        title = data.get('title', video_id)
        return {
            'title': title,
            'author_name': data.get('author_name', ''),
            'upload_date': data.get('upload_date', '')
        }
    except Exception as e:
        print(f"Error getting video info for {video_id}: {str(e)}")
        return {'title': video_id}

def sanitize_filename(title):
    """Convert title to safe filename"""
    # Remove invalid filename characters
    filename = re.sub(r'[<>:"/\\|?*]', '', title)
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    # Ensure the filename isn't too long
    if len(filename) > 200:
        filename = filename[:200]
    return filename.strip('_')

def format_transcript_text(transcript_data):
    """Format transcript data into clean markdown text"""
    # Combine sequential text entries
    full_text = ""
    buffer = ""
    
    for item in transcript_data:
        text = item['text'].strip()
        
        # Detect if it's end of sentence
        if text.endswith(('.', '!', '?')):
            buffer += f"{text} "
            full_text += buffer + "\n\n"
            buffer = ""
        else:
            buffer += f"{text} "
    
    # Add any remaining text
    if buffer:
        full_text += buffer + "\n\n"
    
    # Clean up the text
    # Remove multiple spaces
    full_text = re.sub(r'\s+', ' ', full_text)
    # Remove multiple newlines
    full_text = re.sub(r'\n\s*\n', '\n\n', full_text)
    # Capitalize sentences
    full_text = '. '.join(s.strip().capitalize() for s in full_text.split('. '))
    
    return full_text.strip()

def get_video_ids():
    """Retrieve all video IDs from the channel"""
    print("Fetching video IDs from channel...")
    videos = scrapetube.get_channel(CHANNEL_ID)
    return [video['videoId'] for video in videos]

def get_transcript(video_id):
    """Retrieve transcript for a video in any available language"""
    try:
        # First try to get available transcript languages
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get English transcript first
        try:
            transcript = transcript_list.find_transcript(['en'])
        except:
            # If English is not available, get the first available transcript
            transcript = transcript_list.find_transcript(transcript_list.transcript_data.keys())
        
        return transcript.fetch()
    except Exception as e:
        print(f"Error getting transcript for video {video_id}: {str(e)}")
        return None

def create_markdown_content(video_info, transcript_text):
    """Create a formatted markdown document"""
    markdown = f"""# {video_info['title']}

By {video_info['author_name']}
{video_info.get('upload_date', '')}

## Transcript

{transcript_text}
"""
    return markdown

def save_transcript(video_id, transcript_data, video_info):
    """Save transcript data as a formatted markdown file"""
    # Create filename from video title
    filename = f"{sanitize_filename(video_info['title'])}.md"
    filepath = os.path.join(TRANSCRIPTS_DIR, filename)
    
    # Save the file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(transcript_data)
    
    return filepath

def transcript_exists(video_id, video_info):
    """Check if transcript already exists for the video"""
    title = video_info.get('title', video_id)
    filename = sanitize_filename(title)
    transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{filename}.md")
    return os.path.exists(transcript_path)

def process_video(video_id):
    """Process a single video - get info, transcript, and save"""
    try:
        video_info = get_video_info(video_id)
        
        # Skip if transcript already exists
        if transcript_exists(video_id, video_info):
            return {'status': 'skipped', 'video_id': video_id, 'title': video_info['title']}
        
        transcript_data = get_transcript(video_id)
        if transcript_data:
            transcript_text = format_transcript_text(transcript_data)
            markdown_content = create_markdown_content(video_info, transcript_text)
            save_transcript(video_id, markdown_content, video_info)
            return {'status': 'success', 'video_id': video_id, 'title': video_info['title']}
        return {'status': 'failed', 'video_id': video_id, 'title': video_info['title']}
    except Exception as e:
        return {'status': 'error', 'video_id': video_id, 'error': str(e)}

def main():
    """Main function to process all videos from the channel"""
    ensure_directory_exists(TRANSCRIPTS_DIR)
    
    # Get all video IDs
    video_ids = get_video_ids()
    print(f"Found {len(video_ids)} videos")
    
    # Process videos in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Create a dictionary to store future objects
        future_to_video = {executor.submit(process_video, video_id): video_id 
                          for video_id in video_ids}
        
        # Initialize counters
        processed = 0
        skipped = 0
        failed = 0
        
        # Process results as they complete
        with tqdm(total=len(video_ids), desc="Processing videos") as pbar:
            for future in concurrent.futures.as_completed(future_to_video):
                result = future.result()
                if result['status'] == 'success':
                    processed += 1
                    pbar.set_postfix({'processed': processed, 'skipped': skipped, 'failed': failed})
                elif result['status'] == 'skipped':
                    skipped += 1
                    pbar.set_postfix({'processed': processed, 'skipped': skipped, 'failed': failed})
                else:
                    failed += 1
                    pbar.set_postfix({'processed': processed, 'skipped': skipped, 'failed': failed})
                pbar.update(1)
    
    print(f"\nCompleted processing videos:")
    print(f"- Successfully processed: {processed}")
    print(f"- Skipped (already exist): {skipped}")
    print(f"- Failed: {failed}")

if __name__ == "__main__":
    main()
