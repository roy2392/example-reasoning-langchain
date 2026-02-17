"""
Azure AI Foundry - GPT-5.2 Reasoning Example

This script demonstrates how to use the GPT-5.2 model via Azure AI Foundry
with the Responses API, including:
  1. Reasoning effort control (low / medium / high)
  2. Reasoning summary output (the model's chain-of-thought summary)
  3. Verbosity control (new GPT-5 feature)
  4. Extracting reasoning_tokens from usage details

Prerequisites:
  - An Azure AI Foundry resource with a gpt-5.2 deployment
  - pip install openai python-dotenv
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT"
)  # e.g. https://<resource>.services.ai.azure.com/api/projects/<project>
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2-chat")

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
    raise SystemExit(
        "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in your .env file."
    )

# ---------------------------------------------------------------------------
# Create the client (OpenAI SDK pointing at Azure AI Foundry v1 endpoint)
# The v1 endpoint is at the resource level, not the project level.
# ---------------------------------------------------------------------------
# Strip project path if present to get the resource-level endpoint
resource_endpoint = AZURE_OPENAI_ENDPOINT
if "/api/projects/" in resource_endpoint:
    resource_endpoint = resource_endpoint.split("/api/projects/")[0]

client = OpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    base_url=f"{resource_endpoint.rstrip('/')}/openai/v1/",
)


# ---------------------------------------------------------------------------
# Helper: pretty-print a Responses API result
# ---------------------------------------------------------------------------
def print_response(label: str, response) -> None:
    """Print the model's reasoning summary, answer, and token usage."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    for item in response.output:
        # Reasoning block (contains the chain-of-thought summary)
        if item.type == "reasoning":
            if hasattr(item, "summary") and item.summary:
                print("\n--- Reasoning Summary ---")
                for part in item.summary:
                    if hasattr(part, "text"):
                        print(part.text)
            else:
                print("\n--- Reasoning (no summary returned) ---")

        # Assistant message block (the final answer)
        elif item.type == "message":
            for content_part in item.content:
                if hasattr(content_part, "text"):
                    print(f"\n--- Answer ---\n{content_part.text}")

    # Token usage breakdown
    usage = response.usage
    print(f"\n--- Token Usage ---")
    print(f"  Input tokens:     {usage.input_tokens}")
    print(f"  Output tokens:    {usage.output_tokens}")
    if hasattr(usage, "output_tokens_details") and usage.output_tokens_details:
        reasoning_tokens = getattr(usage.output_tokens_details, "reasoning_tokens", 0)
        print(f"  Reasoning tokens: {reasoning_tokens}")
    print(f"  Total tokens:     {usage.total_tokens}")
    print()


# ---------------------------------------------------------------------------
# Example 1: Math problem with reasoning summary
# ---------------------------------------------------------------------------
def example_basic_reasoning():
    """Send a math problem and get the reasoning summary back."""
    response = client.responses.create(
        model=DEPLOYMENT_NAME,
        input=[
            {
                "role": "developer",
                "content": "You are a helpful math tutor. Show your work step by step.",
            },
            {
                "role": "user",
                "content": (
                    "A train leaves Station A at 9:00 AM traveling at 80 km/h. "
                    "Another train leaves Station B (320 km away) at 10:00 AM "
                    "traveling toward Station A at 120 km/h. "
                    "At what time do they meet, and how far from Station A?"
                ),
            },
        ],
        reasoning={
            "effort": "medium",
            "summary": "detailed",  # auto | detailed
        },
    )
    print_response("Example 1 - Math Problem (reasoning + summary)", response)


# ---------------------------------------------------------------------------
# Example 2: Coding / logic puzzle with reasoning
# ---------------------------------------------------------------------------
def example_coding_reasoning():
    """Ask a coding question that benefits from chain-of-thought reasoning."""
    response = client.responses.create(
        model=DEPLOYMENT_NAME,
        input=[
            {
                "role": "developer",
                "content": "You are a senior software engineer. Think carefully before answering.",
            },
            {
                "role": "user",
                "content": (
                    "Write a Python function that determines if a given string "
                    "of parentheses, brackets, and braces is balanced. "
                    "Explain your approach first, then provide the code."
                ),
            },
        ],
        reasoning={
            "effort": "medium",
            "summary": "detailed",
        },
    )
    print_response("Example 2 - Balanced Brackets (reasoning + summary)", response)


# ---------------------------------------------------------------------------
# Example 3: Raw JSON dump to inspect the full response structure
# ---------------------------------------------------------------------------
def example_raw_response():
    """Dump the full JSON response so you can see every field the API returns."""
    response = client.responses.create(
        model=DEPLOYMENT_NAME,
        input="How many r's are in the word 'strawberry'?",
        reasoning={
            "effort": "medium",
            "summary": "auto",
        },
    )
    print(f"\n{'=' * 60}")
    print("  Example 3 - Raw JSON Response")
    print(f"{'=' * 60}")
    print(response.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Azure AI Foundry - GPT-5.2-chat Reasoning Examples")
    print("=" * 60)

    example_basic_reasoning()
    example_coding_reasoning()
    example_raw_response()

    print("\nDone.")
