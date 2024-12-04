"""Progress tracking utilities for the transcript retriever."""

from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm

@dataclass
class ProgressTracker:
    """Track progress of transcript processing."""
    total: int
    desc: str = 'Processing videos'
    unit: str = 'video'
    
    def __post_init__(self):
        self.progress_bar = tqdm(
            total=self.total,
            desc=self.desc,
            unit=self.unit
        )
        self.processed = 0
        self.errors = 0
        self.skipped = 0
    
    def update(self, status: str, message: Optional[str] = None) -> None:
        """Update progress with status and optional message."""
        if status == 'success':
            self.processed += 1
        elif status == 'error':
            self.errors += 1
        elif status == 'skipped':
            self.skipped += 1
        
        self.progress_bar.update(1)
        if message:
            self.progress_bar.write(message)
    
    def finish(self) -> None:
        """Complete progress tracking and display summary."""
        self.progress_bar.close()
        
        # Print summary
        print('\n' + '='*50)
        print('Processing Summary:')
        print(f'Successfully processed: {self.processed} videos')
        print(f'Skipped (already exists): {self.skipped} videos')
        print(f'Failed to process: {self.errors} videos')
        print('='*50)
