# Transcript Analysis System

A powerful system for extracting health, diet, and longevity insights from transcript files using Google's Gemini models.

## Features

- Parallel processing of multiple transcripts
- Automatic model fallback (gemini-exp-1206 → gemini-exp-1121 → gemini-exp-1114)
- Structured markdown report generation
- Comprehensive error handling and logging
- Configurable processing options

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your API keys:

```
GOOGLE_API_KEY=your_google_api_key_here
```

3. Place your transcript files (markdown format) in the `transcripts` directory

## Usage

Run the analysis:

```bash
python main.py
```

The system will:

1. Process all transcripts in parallel
2. Extract relevant health and longevity insights
3. Generate a comprehensive markdown report in `output/insights_report.md`

## Configuration

Edit `config/config.yaml` to customize:

- Model settings and fallback behavior
- Number of parallel processing threads
- Report sections and format
- Logging level

## Output

The generated report includes sections for:

- Diet Insights
- Supplements
- Scientific Methods
- Health Markers

Each insight is linked to its source transcript and includes the model used for processing.
