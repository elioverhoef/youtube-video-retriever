import logging
from download_transcipts import main as download
from build_report import main as process_report
from process_report import main as process_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_pipeline():
    try:
        # download()
        process_report()
        process_report()
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    run_pipeline()
