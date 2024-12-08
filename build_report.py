from pathlib import Path
import logging
from src.processors.transcript_processor import TranscriptProcessor
from src.processors.parallel_processor import ParallelProcessor
from src.output.report_builder import ReportBuilder
import yaml


def main():
    # Load config
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    # Configure logging
    logging.basicConfig(
        level=config["logging"]["level"],
        format="%(asctime)s�� %(name)s 📝 %(levelname)s 💬 %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        # Initialize components
        processor = TranscriptProcessor()
        parallel_processor = ParallelProcessor()
        report_builder = ReportBuilder()

        # Get all transcript files
        transcript_dir = Path("transcripts")
        transcripts = list(transcript_dir.glob("*.md"))

        if not transcripts:
            logger.error("❌ No transcript files found in ./transcripts directory")
            return

        logger.info(f"📁 Found {len(transcripts)} transcript files")

        # Process transcripts in parallel
        results = parallel_processor.process_transcripts(
            transcripts=transcripts, process_func=processor.process_transcript
        )

        # Filter out None results from failed processing
        results = [r for r in results if r is not None]

        if not results:
            logger.error("❌ No successful results from transcript processing")
            return

        # Build and save report
        report = report_builder.build_report(results)
        report_builder.save_report(report)

        logger.info("✅ Processing completed successfully!")

    except Exception as e:
        logger.error(f"❌ Error in main process: {str(e)}")
        raise


if __name__ == "__main__":
    main()
