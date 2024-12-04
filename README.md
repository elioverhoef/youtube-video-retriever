# YouTube Video Transcript Retriever

This tool retrieves transcripts from all videos on Michael Lustgarden's YouTube channel using `scrapetube` and `youtube-transcript-api`.

## Features

- Automatically fetches all video IDs from the specified YouTube channel
- Retrieves available transcripts for each video
- Saves transcripts in JSON format
- Handles multiple languages (prioritizes English)
- Includes progress bar for tracking
- Skips already downloaded transcripts

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/elioverhoef/youtube-video-retriever.git
   cd youtube-video-retriever
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Simply run the main script:

```bash
python main.py
```

The script will:
1. Create a `transcripts` directory if it doesn't exist
2. Fetch all video IDs from the channel
3. Download available transcripts for each video
4. Save transcripts as JSON files in the `transcripts` directory

## Output

Transcripts are saved in the `transcripts` directory with the video ID as filename:
- `transcripts/[video_id].json`

Each transcript file contains an array of objects with:
- `text`: The transcript text
- `start`: Start time in seconds
- `duration`: Duration in seconds

## Error Handling

- If a video has no available transcripts, it will be skipped
- If a transcript file already exists, it will be skipped
- Errors are logged to the console

## Dependencies

- scrapetube
- youtube_transcript_api
- requests
- tqdm (for progress bar)