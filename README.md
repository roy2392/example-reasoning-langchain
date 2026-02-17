# Azure AI Foundry - GPT-5.2 Reasoning Examples

Python examples demonstrating GPT-5.2 reasoning capabilities via Azure AI Foundry, using both the raw OpenAI SDK and LangChain.

## Features demonstrated

- **Reasoning effort** (`medium`) - controls how long the model "thinks" before answering
- **Reasoning summary** (`auto`, `detailed`) - surfaces the model's chain-of-thought as a summary
- **Token usage breakdown** - shows `reasoning_tokens` vs regular output tokens

## Prerequisites

1. An Azure AI Foundry resource with a **gpt-5.2-chat** model deployment
   (requires [limited access approval](https://aka.ms/oai/gpt5access))
2. Python 3.10+

## Setup

```bash
# Clone and enter the repo
git clone <repo-url>
cd gpt-2.5-examople-reasoning

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Azure endpoint, API key, and deployment name
```

## Run

### OpenAI SDK example

```bash
python main.py
```

Runs three examples using the raw OpenAI Responses API:

| # | Description | Reasoning effort |
|---|-------------|-----------------|
| 1 | Math word problem (train meeting point) | `medium` |
| 2 | Coding puzzle (balanced brackets) | `medium` |
| 3 | Raw JSON response dump (strawberry question) | `medium` |

### LangChain example

```bash
python langchain_example.py
```

Runs three examples using LangChain's `ChatOpenAI` with the Responses API:

| # | Description | Reasoning effort |
|---|-------------|-----------------|
| 1 | Math word problem (train meeting point) | `medium` |
| 2 | Coding puzzle (balanced brackets) | `medium` |
| 3 | Strawberry question with full metadata dump | `medium` |

## Key API patterns

### OpenAI SDK - Responses API with reasoning

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-key",
    base_url="https://<resource>.services.ai.azure.com/openai/v1/",
)

response = client.responses.create(
    model="gpt-5.2-chat",
    input=[{"role": "user", "content": "your prompt"}],
    reasoning={
        "effort": "medium",     # only "medium" is supported
        "summary": "detailed",  # auto | detailed
    },
)
```

### LangChain - ChatOpenAI with reasoning

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-5.2-chat",
    api_key="your-key",
    base_url="https://<resource>.services.ai.azure.com/openai/v1/",
    reasoning={"effort": "medium", "summary": "auto"},
)

response = llm.invoke([HumanMessage(content="your prompt")])
```

### Reading reasoning tokens from usage

```python
# OpenAI SDK
reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens

# LangChain
usage = response.response_metadata["token_usage"]
reasoning_tokens = usage["completion_tokens_details"]["reasoning_tokens"]
```

## Important notes

- Use `ChatOpenAI` (not `AzureChatOpenAI`) with `base_url` pointing to the resource-level `/openai/v1/` endpoint.
- The deployed model name is `gpt-5.2-chat` (not `gpt-5.2` or the full date-suffixed name).
- Only `reasoning_effort="medium"` is supported. Using `"low"` or `"high"` returns an error.
- Reasoning summaries are not guaranteed on every request -- this is expected behavior.
- The `reasoning_tokens` field shows hidden tokens the model used internally; they count toward billing but are not shown in the response text.
- Do not use `temperature`, `top_p`, or other sampling parameters with reasoning models -- they are not supported.

## References

- [Azure OpenAI reasoning models](https://learn.microsoft.com/azure/ai-foundry/openai/how-to/reasoning?view=foundry-classic)
- [Responses API](https://learn.microsoft.com/azure/ai-foundry/openai/how-to/responses?view=foundry-classic)
- [LangChain ChatOpenAI with reasoning](https://docs.langchain.com/oss/python/integrations/chat/openai#using-with-azure-openai)
- [GPT-5 prompting cookbook](https://cookbook.openai.com/examples/gpt-5/gpt-5_prompting_guide)
