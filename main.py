import os
import json
import scrapetube
from youtube_transcript_api import YouTubeTranscriptApi
from tqdm import tqdm

# Constants
CHANNEL_ID = "UCT1UMLpZ_CrQ_8I431K0b-g"  # Michael Lustgarden's channel ID
TRANSCRIPTS_DIR = "transcripts"

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

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

def save_transcript(video_id, transcript_data):
    """Save transcript data to a JSON file"""
    if transcript_data:
        filename = os.path.join(TRANSCRIPTS_DIR, f"{video_id}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=2)

def main():
    # Ensure transcripts directory exists
    ensure_directory_exists(TRANSCRIPTS_DIR)
    
    # Get all video IDs
    video_ids = get_video_ids()
    print(f"Found {len(video_ids)} videos")
    
    # Process each video
    for video_id in tqdm(video_ids, desc="Processing videos"):
        # Skip if transcript already exists
        transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{video_id}.json")
        if os.path.exists(transcript_path):
            continue
            
        # Get and save transcript
        transcript = get_transcript(video_id)
        if transcript:
            save_transcript(video_id, transcript)

if __name__ == "__main__":
    main()