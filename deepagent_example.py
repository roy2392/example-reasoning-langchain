import json
import os

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

# Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2-chat")

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
    raise SystemExit(
        "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in your .env file."
    )

# Build the base_url 
resource_endpoint = AZURE_OPENAI_ENDPOINT
if "/api/projects/" in resource_endpoint:
    resource_endpoint = resource_endpoint.split("/api/projects/")[0]

base_url = f"{resource_endpoint.rstrip('/')}/openai/v1/"

# Create the LangChain ChatOpenAI model with reasoning enabled
llm = ChatOpenAI(
    model=DEPLOYMENT_NAME,
    api_key=SecretStr(AZURE_OPENAI_API_KEY),
    base_url=base_url,
    reasoning={"effort": "medium", "summary": "auto"},
)

# Custom tools for the agent
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Args:
        expression: A Python math expression to evaluate (e.g. '2 + 3 * 4').
    """
    try:
        # Only allow safe math operations
        allowed_names = {"__builtins__": {}}
        result = eval(expression, allowed_names)  # noqa: S307
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


def count_characters(text: str, character: str) -> str:
    """Count how many times a specific character appears in a text.

    Args:
        text: The text to search in.
        character: The single character to count.
    """
    count = text.lower().count(character.lower())
    return f"The character '{character}' appears {count} time(s) in '{text}'."

# Helper: print the agent result
def print_result(label: str, result: dict) -> None:
    """Print the final agent response and token usage."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    messages = result.get("messages", [])
    if not messages:
        print("  (no messages returned)")
        return

    # The last message is the agent's final answer
    final_msg = messages[-1]

    # Print reasoning summary if present (content is a list of blocks)
    if isinstance(final_msg.content, list):
        for block in final_msg.content:
            if isinstance(block, dict):
                if block.get("type") == "reasoning":
                    for summary_part in block.get("summary", []):
                        text = summary_part.get("text", "")
                        if text:
                            print(f"\n--- Reasoning Summary ---\n{text}")
                elif block.get("type") == "text":
                    print(f"\n--- Answer ---\n{block.get('text', '')}")
    elif isinstance(final_msg.content, str) and final_msg.content:
        print(f"\n--- Answer ---\n{final_msg.content}")

    # Token usage
    usage = getattr(final_msg, "usage_metadata", None)
    if usage:
        print(f"\n--- Token Usage (final message) ---")
        print(f"  Input tokens:     {usage.get('input_tokens', 'N/A')}")
        print(f"  Output tokens:    {usage.get('output_tokens', 'N/A')}")
        details = usage.get("output_token_details", {})
        if details and details.get("reasoning") is not None:
            print(f"  Reasoning tokens: {details['reasoning']}")
        print(f"  Total tokens:     {usage.get('total_tokens', 'N/A')}")

    # Show tool calls made during the conversation
    tool_messages = [m for m in messages if getattr(m, "type", "") == "tool"]
    if tool_messages:
        print(f"\n--- Tools Used ({len(tool_messages)} call(s)) ---")
        for tm in tool_messages:
            print(f"  [{tm.name}] -> {tm.content[:120]}")

    print()


# Example 1: Math problem with tool use
def example_math_with_tools():
    """Agent uses the calculate tool alongside reasoning."""
    agent = create_deep_agent(
        model=llm,
        tools=[calculate],
        system_prompt=(
            "You are a helpful math tutor. When you need to compute a "
            "numerical result, use the 'calculate' tool. Show your reasoning."
        ),
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "A train leaves Station A at 9:00 AM traveling at 80 km/h. "
                        "Another train leaves Station B (320 km away) at 10:00 AM "
                        "traveling toward Station A at 120 km/h. "
                        "At what time do they meet, and how far from Station A? "
                        "Use the calculate tool to verify your math."
                    ),
                }
            ]
        }
    )
    print_result("Example 1 - Math + Tool Use (Deep Agent)", result)


# Main
if __name__ == "__main__":
    print("Azure AI Foundry - GPT-5.2-chat Reasoning Examples (Deep Agent)")
    print("=" * 60)

    example_math_with_tools()
    example_character_counting()
    example_pure_reasoning()

    print("\nDone.")
