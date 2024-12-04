# YouTube Video Transcript Retriever

This tool retrieves transcripts from all videos on Michael Lustgarden's YouTube channel using `scrapetube` and `youtube-transcript-api`, formatting them into readable Markdown documents.

## Features

- Automatically fetches all video IDs from the specified YouTube channel
- Retrieves available transcripts for each video
- Gets video metadata (title, author, upload date)
- Formats transcripts into clean, readable text
- Saves transcripts as formatted Markdown files
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
3. Download available transcripts and video metadata
4. Format the transcripts into readable text
5. Save transcripts as Markdown files in the `transcripts` directory

## Output

Transcripts are saved in the `transcripts` directory with the video title as filename:
- `transcripts/[video_title].md`

Each Markdown file contains:
- Video title
- Author name
- Upload date (if available)
- Formatted transcript text
- Footer with source attribution

## Text Formatting

The transcript text is automatically formatted to:
- Combine sequential text entries into proper sentences
- Add appropriate line breaks
- Capitalize sentences
- Remove redundant spaces and newlines
- Create a clean, readable document

## Error Handling

- If a video has no available transcripts, it will be skipped
- If a transcript file already exists, it will be skipped
- Errors are logged to the console

## Dependencies

- scrapetube
- youtube_transcript_api
- requests
- tqdm (for progress bar)