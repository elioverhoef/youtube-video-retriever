from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import yaml
import logging
from typing import List, Callable, Any
from tqdm import tqdm


class ParallelProcessor:
    def __init__(self):
        # Load config
        config_path = Path("config/config.yaml")
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.num_threads = self.config["processing"]["num_threads"]
        self.logger = logging.getLogger(__name__)

    def process_transcripts(
        self, transcripts: List[Path], process_func: Callable[[Path], Any]
    ) -> List[Any]:
        """Process multiple transcripts in parallel using ThreadPoolExecutor.

        Args:
            transcripts: List of transcript file paths
            process_func: Function that processes a single transcript

        Returns:
            List of results from processing each transcript
        """
        results = []

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Create progress bar
            futures = []
            for transcript in transcripts:
                future = executor.submit(process_func, transcript)
                futures.append(future)

            # Process results as they complete
            for future in tqdm(
                futures, desc="Processing transcripts", total=len(transcripts)
            ):
                try:
                    result = future.result()
                    if result:
                        results.extend(result if isinstance(result, list) else [result])
                except Exception as e:
                    self.logger.error(f"Error processing transcript: {str(e)}")

        return results
