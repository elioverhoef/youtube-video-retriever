"""Custom exceptions for the YouTube Transcript Retriever."""

class YouTubeTranscriptError(Exception):
    """Base exception for YouTube Transcript Retriever."""
    pass

class ChannelNotFoundError(YouTubeTranscriptError):
    """Raised when a YouTube channel cannot be found."""
    pass

class VideoNotFoundError(YouTubeTranscriptError):
    """Raised when a YouTube video cannot be found."""
    pass

class TranscriptNotFoundError(YouTubeTranscriptError):
    """Raised when no transcript is available for a video."""
    pass

class InvalidChannelURLError(YouTubeTranscriptError):
    """Raised when the provided channel URL is invalid."""
    pass