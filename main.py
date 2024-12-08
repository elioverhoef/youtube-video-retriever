import logging
from typing import List, Optional
from enum import Enum, auto

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Step(Enum):
    DOWNLOAD = auto()
    BUILD = auto()
    PROCESS = auto()


def run_download() -> bool:
    """Run the transcript download step"""
    try:
        from download_transcipts import main as download_main

        download_main()
        return True
    except Exception as e:
        logging.error(f"Failed to download transcripts: {e}")
        return False


def run_build() -> bool:
    """Run the report building step"""
    try:
        from build_report import main as build_main

        build_main()
        return True
    except Exception as e:
        logging.error(f"Failed to build report: {e}")
        return False


def run_process() -> bool:
    """Run the report processing step"""
    try:
        from process_report import main as process_main

        process_main()
        return True
    except Exception as e:
        logging.error(f"Failed to process report: {e}")
        return False


def run_pipeline(steps: Optional[List[Step]] = None) -> bool:
    """Run specified steps or all steps if none specified"""
    steps = steps or [Step.DOWNLOAD, Step.BUILD, Step.PROCESS]

    step_functions = {
        Step.DOWNLOAD: run_download,
        Step.BUILD: run_build,
        Step.PROCESS: run_process,
    }

    for step in steps:
        logging.info(f"Starting step: {step.name}")
        if not step_functions[step]():
            logging.error(f"Pipeline failed at step: {step.name}")
            return False
        logging.info(f"Completed step: {step.name}")

    return True


def main():
    import sys

    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            steps = [Step[arg.upper()] for arg in sys.argv[1:]]
        except KeyError as e:
            print(f"Invalid step: {e}")
            print("Valid steps: download, build, process")
            return
    else:
        steps = None

    # Run pipeline
    success = run_pipeline(steps)

    if success:
        logging.info("Pipeline completed successfully!")
    else:
        logging.error("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
