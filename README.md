# 🚀 YouTube Video Transcript Retriever

A blazing fast, parallel YouTube transcript retriever that downloads and formats transcripts from any YouTube channel.

## ✨ Features

- 🔄 Parallel processing for super fast transcript retrieval
- 🎯 Works with any YouTube channel (just provide the URL)
- 📂 Organizes transcripts by channel name
- 🌍 Handles multiple languages (prioritizes English)
- 📊 Shows progress with detailed logging
- ⚡ Skips already downloaded transcripts
- 🎨 Formats transcripts into clean, readable Markdown
- 🛡️ Robust error handling and recovery

## 🛠️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/elioverhoef/youtube-video-retriever.git
   cd youtube-video-retriever
   ```

2. Install using pip:
   ```bash
   pip install -e .
   ```

## 📖 Usage

### Command Line Interface

The simplest way to use the tool is through its command-line interface:

```bash
youtube-transcripts "https://www.youtube.com/channel/YOUR_CHANNEL_ID"
```

Or using the channel ID directly:

```bash
youtube-transcripts YOUR_CHANNEL_ID
```

### Command Line Options

```
usage: youtube-transcripts [-h] [-o OUTPUT_DIR] [-w WORKERS] [-v] channel_url

Retrieve and process YouTube video transcripts

positional arguments:
  channel_url           YouTube channel URL or ID

optional arguments:
  -h, --help           show this help message and exit
  -o OUTPUT_DIR        Output directory for transcripts (default: transcripts)
  -w WORKERS           Number of worker processes (default: 4)
  -v, --verbose        Enable verbose logging
```

### Python API

You can also use the tool programmatically:

```python
from youtube_transcript_retriever.core.channel import ChannelProcessor

# Create a processor instance
processor = ChannelProcessor(
    channel_url="https://www.youtube.com/channel/YOUR_CHANNEL_ID",
    output_dir="transcripts",
    workers=4  # Number of parallel workers
)

# Process all videos in the channel
processor.process_channel()
```

## 📁 Output Structure

Transcripts are organized by channel name:

```
transcripts/
└── Channel_Name/
    ├── Video_Title_1.md
    ├── Video_Title_2.md
    └── ...
```

Each Markdown file contains:
- Video title
- Author name
- Upload date (if available)
- Formatted transcript text
- Footer with source attribution

## 🎯 Text Processing Features

The transcript formatter:
- Combines sequential text entries into proper sentences
- Adds appropriate line breaks
- Capitalizes sentences
- Removes redundant spaces and newlines
- Creates clean, readable documents

## ⚙️ Technical Details

- Uses `scrapetube` for channel video discovery
- `youtube-transcript-api` for transcript retrieval
- Process pool for parallel processing
- Type hints throughout the codebase
- Modular design for easy extension
- Comprehensive error handling

## 🐛 Error Handling

The tool handles various error cases:
- Invalid channel URLs/IDs
- Missing transcripts
- Network issues
- Rate limiting
- File system errors

Failed video processing won't stop the entire operation - the tool will continue with remaining videos.

## 📋 Requirements

- Python 3.9+
- scrapetube
- youtube-transcript-api
- requests
- tqdm

## 💡 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
