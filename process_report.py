from src.models.gemini_client import GeminiClient
from src.processors.insight_processor import ReportProcessor

def main():
    # Initialize GeminiClient first
    gemini_client = GeminiClient()
    
    # Pass gemini_client to ReportProcessor
    processor = ReportProcessor(gemini_client)
    
    # Process the report
    input_path = "output/insights_report.md"
    output_path = "output/processed_insights_report.md"
    
    print("Starting report processing...")
    processor.process_report(input_path, output_path)
    print(f"Processing complete! New report saved to: {output_path}")

if __name__ == "__main__":
    main()
