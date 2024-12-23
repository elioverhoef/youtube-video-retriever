import os
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Tuple


def test_api_key(api_key: str) -> Tuple[bool, str]:
    """Test a single Gemini API key.

    Returns:
        Tuple[bool, str]: (success status, error message if any)
    """
    try:
        # Configure the API
        genai.configure(api_key=api_key)

        # Initialize model
        model = genai.GenerativeModel("gemini-exp-1206")

        # Simple yes/no test prompt
        response = model.generate_content(
            "Please answer with a single word: Is the sky blue?"
        )

        # If we get here, the API call was successful
        return True, f"Success - Response: {response.text}"

    except Exception as e:
        return False, str(e)


def main():
    # Load environment variables
    load_dotenv()

    # Get all Google API keys from environment
    api_keys = {
        key: os.getenv(key) for key in os.environ if key.startswith("GOOGLE_API_KEY")
    }

    # Test each key
    print("Testing Gemini API keys...")
    print("-" * 50)

    for key_name, api_key in api_keys.items():
        print(f"\nTesting {key_name}:")
        success, message = test_api_key(api_key)
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"Status: {status}")
        print(f"Details: {message}")


if __name__ == "__main__":
    main()
