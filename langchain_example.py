"""
Azure AI Foundry - GPT-5.2 Reasoning Example (LangChain)

This script demonstrates how to use GPT-5.2 reasoning capabilities via
Azure AI Foundry using LangChain's ChatOpenAI, including:
  1. Reasoning effort control (medium)
  2. Reasoning summary output (the model's chain-of-thought summary)
  3. Extracting reasoning_tokens from usage details

Prerequisites:
  - An Azure AI Foundry resource with a gpt-5.2-chat deployment
  - pip install langchain-openai python-dotenv
"""

import json
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2-chat")

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
    raise SystemExit(
        "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in your .env file."
    )

# ---------------------------------------------------------------------------
# Build the base_url (resource-level v1 endpoint)
# ---------------------------------------------------------------------------
resource_endpoint = AZURE_OPENAI_ENDPOINT
if "/api/projects/" in resource_endpoint:
    resource_endpoint = resource_endpoint.split("/api/projects/")[0]

base_url = f"{resource_endpoint.rstrip('/')}/openai/v1/"

# ---------------------------------------------------------------------------
# Create the LangChain ChatOpenAI client
#
# Key points:
#   - Use ChatOpenAI (NOT AzureChatOpenAI) with base_url for Azure AI Foundry v1
#   - Pass reasoning={"effort": "medium", "summary": "auto"} to enable
#     the Responses API with chain-of-thought reasoning
#   - gpt-5.2-chat only supports reasoning_effort="medium"
# ---------------------------------------------------------------------------
llm = ChatOpenAI(
    model=DEPLOYMENT_NAME,
    api_key=SecretStr(AZURE_OPENAI_API_KEY),
    base_url=base_url,
    reasoning={"effort": "medium", "summary": "auto"},
)


# ---------------------------------------------------------------------------
# Helper: pretty-print a LangChain AIMessage with reasoning
# ---------------------------------------------------------------------------
def print_response(label: str, response) -> None:
    """Print the model's reasoning summary, answer, and token usage.

    When reasoning is enabled, response.content is a list of dicts
    (not a plain string). Each dict has a "type" field:
      - "reasoning": contains a "summary" list with chain-of-thought summaries
      - "text":      contains the final answer in "text"
    Token usage lives in response.usage_metadata (not response_metadata).
    """
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    # response.content is a list of content blocks when reasoning is enabled
    if isinstance(response.content, list):
        for block in response.content:
            if not isinstance(block, dict):
                continue

            # Reasoning block: extract summary text
            if block.get("type") == "reasoning":
                for summary_part in block.get("summary", []):
                    text = summary_part.get("text", "")
                    if text:
                        print(f"\n--- Reasoning Summary ---\n{text}")

            # Text block: the final answer
            elif block.get("type") == "text":
                print(f"\n--- Answer ---\n{block.get('text', '')}")

    elif isinstance(response.content, str) and response.content:
        # Fallback for non-reasoning responses (plain string)
        print(f"\n--- Answer ---\n{response.content}")

    # Token usage from usage_metadata (LangChain's standard location)
    usage = response.usage_metadata
    if usage:
        print(f"\n--- Token Usage ---")
        print(f"  Input tokens:     {usage.get('input_tokens', 'N/A')}")
        print(f"  Output tokens:    {usage.get('output_tokens', 'N/A')}")
        details = usage.get("output_token_details", {})
        if details and details.get("reasoning") is not None:
            print(f"  Reasoning tokens: {details['reasoning']}")
        print(f"  Total tokens:     {usage.get('total_tokens', 'N/A')}")
    print()


# ---------------------------------------------------------------------------
# Example 1: Math problem with reasoning summary
# ---------------------------------------------------------------------------
def example_basic_reasoning():
    """Send a math problem and get the reasoning summary back."""
    messages = [
        SystemMessage(
            content="You are a helpful math tutor. Show your work step by step."
        ),
        HumanMessage(
            content=(
                "A train leaves Station A at 9:00 AM traveling at 80 km/h. "
                "Another train leaves Station B (320 km away) at 10:00 AM "
                "traveling toward Station A at 120 km/h. "
                "At what time do they meet, and how far from Station A?"
            ),
        ),
    ]
    response = llm.invoke(messages)
    print_response("Example 1 - Math Problem (reasoning + summary)", response)


# ---------------------------------------------------------------------------
# Example 2: Coding / logic puzzle with reasoning
# ---------------------------------------------------------------------------
def example_coding_reasoning():
    """Ask a coding question that benefits from chain-of-thought reasoning."""
    messages = [
        SystemMessage(
            content="You are a senior software engineer. Think carefully before answering."
        ),
        HumanMessage(
            content=(
                "Write a Python function that determines if a given string "
                "of parentheses, brackets, and braces is balanced. "
                "Explain your approach first, then provide the code."
            ),
        ),
    ]
    response = llm.invoke(messages)
    print_response("Example 2 - Balanced Brackets (reasoning + summary)", response)


# ---------------------------------------------------------------------------
# Example 3: Simple question to show reasoning metadata
# ---------------------------------------------------------------------------
def example_simple_question():
    """Ask a simple question and display the full response metadata."""
    messages = [
        HumanMessage(content="How many r's are in the word 'strawberry'?"),
    ]
    response = llm.invoke(messages)
    print_response("Example 3 - Strawberry Question (reasoning metadata)", response)

    # Also dump usage_metadata and response_metadata for inspection
    print("--- Full usage_metadata ---")
    print(json.dumps(response.usage_metadata, indent=2, default=str))
    print()
    print("--- Full response_metadata ---")
    print(json.dumps(response.response_metadata, indent=2, default=str))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Azure AI Foundry - GPT-5.2-chat Reasoning Examples (LangChain)")
    print("=" * 60)

    example_basic_reasoning()
    example_coding_reasoning()
    example_simple_question()

    print("\nDone.")
