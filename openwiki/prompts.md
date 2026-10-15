---
type: "Concept"
title: "Prompt Templates and Variable Composition"
description: "Prompt templates construct message sequences with variable substitution, supporting single-string and multi-message chat formats. Templates integrate with models via the Runnable interface and support composition via the + operator."
tags: [prompt, template, few-shot, example-selection, variable-substitution, structured-output, composition, runnable]
sources:
  - id: openwiki-source-1f4e0a5b877db4f050f2a34c
    resource: repo://libs/core/langchain_core/example_selectors/base.py
  - id: openwiki-source-d533a177a8d9a5dd46f561d9
    resource: repo://libs/core/langchain_core/example_selectors/length_based.py
  - id: openwiki-source-5e027af8cc764d2750129cf1
    resource: repo://libs/core/langchain_core/example_selectors/semantic_similarity.py
  - id: openwiki-source-03d7415879ed05a392edd62d
    resource: repo://libs/core/langchain_core/prompts/base.py
  - id: openwiki-source-15fdd645c1ee76ae559799c1
    resource: repo://libs/core/langchain_core/prompts/chat.py
  - id: openwiki-source-bc32774051e0e8a931a6fecd
    resource: repo://libs/core/langchain_core/prompts/few_shot.py
  - id: openwiki-source-5549894302ea4dfd5b8f4278
    resource: repo://libs/core/langchain_core/prompts/prompt.py
  - id: openwiki-source-cf81d0ba0a387a7cd9b5dfb8
    resource: repo://libs/core/langchain_core/prompts/string.py
  - id: openwiki-source-204b5e61a019044332bd2dd4
    resource: repo://libs/core/langchain_core/prompts/structured.py
generated: { by: "openwiki/0.5.0", at: "2026-10-15T08:27:20.856Z" }
verified:
  - by: openwiki/0.5.0
    at: 2026-10-15T08:27:20.856Z
---

## Overview

LangChain's **prompt templating system** provides a flexible, composable way to construct messages for language models. All prompt templates inherit from `BasePromptTemplate`, which implements the `RunnableSerializable` interface, enabling seamless integration with chains, async/await patterns, and observability hooks. Prompts accept input variables, format them into message sequences, and optionally parse structured output. The system distinguishes between **string templates** (for raw text) and **chat templates** (sequences of typed messages with explicit roles). **Few-shot prompt templates** add the capability to select and inject examples dynamically, teaching models by demonstration. Composition via the `+` operator merges templates and variables, creating complex nested prompts.

## Fundamental Concepts

### Prompt Types

LangChain provides two main categories of prompts:

#### PromptTemplate (StringPromptTemplate)

A `PromptTemplate` wraps a single string template with variable placeholders. The template is formatted using one of three engines:

- **f-string** (default): Python f-string syntax. Fast, supports arbitrary expressions in `{...}` brackets with proper escaping via `{{` and `}}`.
- **mustache**: Mustache syntax using `{{variable}}`. Safer for user-controlled templates.
- **jinja2**: Full Jinja2 templating. Supports conditionals, loops, and filters, but poses security risks if templates come from untrusted sources; LangChain uses `SandboxedEnvironment` by default for defense-in-depth.

**Key properties:**
- `template`: The template string.
- `input_variables`: List of variable names that must be provided during formatting.
- `partial_variables`: Pre-filled variables; reduce required inputs when formatting.
- `template_format`: Which engine to use (`f-string`, `mustache`, or `jinja2`).

```python
from langchain_core.prompts import PromptTemplate

# Simple f-string prompt
prompt = PromptTemplate.from_template("Tell me about {topic}")
output = prompt.format(topic="machine learning")

# Jinja2 with conditionals
prompt = PromptTemplate(
    template="{% if detailed %}Detailed:{% endif %} {content}",
    template_format="jinja2",
    input_variables=["content"],
    partial_variables={"detailed": True}
)

# Partial variables reduce required inputs
prompt = PromptTemplate(
    template="User: {name}, Topic: {topic}",
    input_variables=["topic"],
    partial_variables={"name": "Alice"}
)
result = prompt.format(topic="AI")  # name is already set
```

#### ChatPromptTemplate

A `ChatPromptTemplate` sequences message prompt templates into a conversation structure. Each message has a role (system, human, ai, tool, etc.) and content. This aligns with the message-based API of chat models like GPT-4 and Claude.

**Constructor patterns:**

```python
from langchain_core.prompts import ChatPromptTemplate

# Tuple shorthand
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Hello, {name}"),
    ("ai", "Hi {name}! How can I help?"),
    ("human", "{user_input}"),
])

# Or direct list construction
template = ChatPromptTemplate([
    ("system", "You are a helpful assistant."),
    ("human", "Hello, {name}"),
])
```

Supported message types in the shorthand syntax:
- `"system"` → `SystemMessagePromptTemplate`
- `"human"` → `HumanMessagePromptTemplate`
- `"ai"` → `AIMessagePromptTemplate`
- `"user"` → Alias for `"human"`
- `"assistant"` → Alias for `"ai"`
- `"tool"` / `"function"` → `ToolMessagePromptTemplate` / `FunctionMessagePromptTemplate`
- `"placeholder"` → `MessagesPlaceholder` for dynamic message lists

**Key methods:**
- `format_messages(**kwargs)`: Returns a list of `BaseMessage` objects.
- `invoke(dict)`: Runnable interface, returns `ChatPromptValue` containing formatted messages.
- `format(**kwargs)`: Converts message list to a single string (useful for debugging or non-chat APIs).

#### MessagesPlaceholder

A `MessagesPlaceholder` injects a pre-formatted list of messages at a specific point in the prompt. This is essential for conversation history, where messages are passed as-is without templating. Unlike other message templates, `MessagesPlaceholder` accepts raw message tuples, BaseMessage instances, or lists thereof.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{question}"),
])

# Pass conversation history as tuples or BaseMessage objects
result = template.invoke({
    "chat_history": [
        ("human", "What's 2+2?"),
        ("ai", "4"),
    ],
    "question": "And 3+3?",
})
# Messages: [SystemMessage, HumanMessage, AIMessage, HumanMessage]
```

**Optional Placeholders:**

The `optional=True` flag makes the placeholder optional, automatically populated with an empty list if not provided:

```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
])

# chat_history not required
result = template.invoke({"input": "Hello"})
# MessagesPlaceholder injects empty list, system and human messages follow
```

**Message Limiting:**

The `n_messages` parameter limits how many recent messages are included, useful for staying within token budgets:

```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    MessagesPlaceholder("chat_history", n_messages=5),  # Keep only 5 recent
    ("human", "{input}"),
])

# If chat_history has 10 messages, only the last 5 are included
result = template.invoke({
    "chat_history": messages_10_long,
    "input": "Hello",
})
```

**Integration with ChatPromptTemplate:**

When a `ChatPromptTemplate` contains an optional `MessagesPlaceholder`, it is automatically added to `optional_variables` and its partial default (empty list) is set. This allows the entire template to be valid even when the placeholder is missing from input.

### Template Variable Substitution and Validation

Prompts automatically detect variable names from template syntax during initialization (via `get_template_variables()` for f-string and mustache, or Jinja2's AST for Jinja2). Required input variables are determined at construction time and validated against provided arguments at format/invoke time.

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Q: {question}\nA: {answer}")
print(prompt.input_variables)  # ['answer', 'question'] (sorted)

# Variables are validated at runtime; missing variables raise KeyError
try:
    prompt.format(question="What?")  # Missing 'answer'
except KeyError as e:
    print(f"Error: {e}")  # "answer"
```

**Input Schema Inference:**

Prompts implement `get_input_schema()` to generate a Pydantic model representing their input contract. This enables:
- Runtime input validation in agents and APIs
- Documentation generation (OpenAPI specs, CLI help)
- IDE autocomplete for prompt parameters

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("human", "{question}"),
    MessagesPlaceholder("chat_history", optional=True),
])

schema = template.get_input_schema()
# Returns a Pydantic model with:
#   question: str (required)
#   chat_history: list[BaseMessage] (optional, defaults to [])
```

**Partial Variable Binding:**

Partial application pre-fills some variables, reducing the required input set. `partial()` returns a new immutable instance without mutating the original:

```python
prompt = PromptTemplate.from_template("User: {name}, Question: {question}")
partial_prompt = prompt.partial(name="Bob")  # Returns new PromptTemplate
output = partial_prompt.format(question="How are you?")

# Original is unchanged
print(prompt.input_variables)      # ['name', 'question']
print(partial_prompt.input_variables)  # ['question']
```

**Single-Variable Convenience:**

When a prompt or ChatPromptTemplate has exactly **one** input variable, `invoke()` accepts a scalar value directly (not wrapped in a dict):

```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are a bot."),
    ("human", "{input}"),
])
result = template.invoke("Hello!")  # Auto-wraps as {"input": "Hello!"}

# Equivalent to:
result = template.invoke({"input": "Hello!"})
```

## Few-Shot Prompt Templates

Few-shot learning teaches models by providing input-output examples before the user's actual query. LangChain provides two patterns: one for string prompts and one for chat-based prompts.

### FewShotPromptTemplate

`FewShotPromptTemplate` formats examples into a single string prompt.

**Structure:**
```
[prefix]

[formatted example 1]

[formatted example 2]

...

[suffix]
```

**Components:**
- `prefix`: Text before examples (optional).
- `example_prompt`: A `PromptTemplate` specifying how each example is formatted.
- `examples` or `example_selector`: Source of examples (either a fixed list or dynamic selector).
- `suffix`: Text after examples. Usually contains the actual task and placeholders for the new input.
- `example_separator`: String joining prefix, examples, and suffix (default: `"\n\n"`).

```python
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
]

example_prompt = PromptTemplate(
    template="Input: {input}\nOutput: {output}",
    input_variables=["input", "output"],
)

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Input: {input}\nOutput:",
    input_variables=["input"],
)

output = prompt.format(input="big")
# Output:
# Input: happy
# Output: sad
#
# Input: tall
# Output: short
#
# Input: big
# Output:
```

### FewShotChatMessagePromptTemplate

`FewShotChatMessagePromptTemplate` embeds examples as message pairs within a chat sequence.

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "What is {input}?"),
    ("ai", "{output}"),
])

few_shot = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful math tutor."),
    few_shot,
    ("human", "What is {input}?"),
])

result = template.invoke({"input": "4+4"})
# Messages: [system, human(2+2?), ai(4), human(2+3?), ai(5), human(4+4?)]
```

## Example Selectors

Instead of using a fixed list of examples, an **example selector** dynamically picks relevant examples based on the input. This optimizes prompt length and relevance.

### BaseExampleSelector Interface

All selectors implement:

```python
class BaseExampleSelector:
    def add_example(self, example: dict[str, str]) -> Any:
        """Add a new example to the store."""
        
    def select_examples(self, input_variables: dict[str, str]) -> list[dict[str, Any]]:
        """Select which examples to use based on inputs."""
```

### SemanticSimilarityExampleSelector

Embeds examples and input into a vector space, retrieving the `k` most similar examples. Requires a `VectorStore` and embeddings model.

```python
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.embeddings import OpenAIEmbeddings
from langchain_core.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
]

# Create vector store from example texts
to_vectorize = [" ".join(example.values()) for example in examples]
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)

selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,  # Always return 2 examples
)

example_prompt = PromptTemplate(
    template="Input: {input}\nOutput: {output}",
    input_variables=["input", "output"],
)

prompt = FewShotPromptTemplate(
    example_selector=selector,
    example_prompt=example_prompt,
    suffix="Input: {input}\nOutput:",
    input_variables=["input"],
)

# When formatting, the selector retrieves the 2 most similar examples to "bright"
output = prompt.format(input="bright")
```

**Key parameters:**
- `vectorstore`: VectorStore containing embedded examples.
- `k`: Number of examples to return (default: 4).
- `input_keys`: Optional filter to use only specific keys for similarity search (e.g., only the "input" field, not "output").
- `example_keys`: Optional filter to include only certain keys in returned examples.
- `vectorstore_kwargs`: Extra arguments passed to the vectorstore's `similarity_search` method.

### LengthBasedExampleSelector

Selects examples greedily up to a maximum token/word count, preventing prompt length overflow. Useful when token budgets are tight.

```python
from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
]

example_prompt = PromptTemplate(
    template="Input: {input}\nOutput: {output}",
    input_variables=["input", "output"],
)

selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=50,  # Limit prompt to ~50 words
    get_text_length=lambda x: len(x.split()),  # Custom length function
)

prompt = FewShotPromptTemplate(
    example_selector=selector,
    example_prompt=example_prompt,
    suffix="Input: {input}\nOutput:",
    input_variables=["input"],
)

# Selector returns only as many examples as fit within max_length
output = prompt.format(input="fast")
```

**Key parameters:**
- `examples`: List of all available examples.
- `max_length`: Maximum prompt length (tokens or words, determined by `get_text_length`).
- `get_text_length`: Function to measure prompt length; defaults to word count via regex.

**Behavior:** Examples are iterated in order; the selector stops adding when the next example would exceed `max_length`. This is greedy, not optimal, but fast and predictable.

## Structured Output Prompts

The `StructuredPrompt` (beta) combines a `ChatPromptTemplate` with a Pydantic schema, enabling the model to produce JSON output matching a specific schema.

```python
from pydantic import BaseModel
from langchain_core.prompts import StructuredPrompt

class QuestionAnswer(BaseModel):
    question: str
    answer: str

template = StructuredPrompt.from_messages_and_schema(
    messages=[
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
    ],
    schema=QuestionAnswer,
)

# When invoked with a model supporting structured output,
# the model is instructed to return JSON matching QuestionAnswer
result = template.invoke({"input": "What is LangChain?"})
```

This is useful for tasks requiring consistent, parseable output (e.g., fact extraction, data classification).

## Runnable Interface and Chaining

All prompts inherit from `RunnableSerializable`, implementing the Runnable interface. This enables prompts to:
- Compose with other Runnable components (models, parsers, chains) via the `|` (pipe) operator
- Support both synchronous and asynchronous execution
- Participate in streaming and batch operations
- Integrate with observability and tracing infrastructure
- Accept configuration dictionaries (run_id, tags, metadata) for debugging and monitoring

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

template = ChatPromptTemplate.from_messages([
    ("system", "You are a poet."),
    ("human", "Write a poem about {topic}"),
])

model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = template | model | parser

# Synchronous
result = chain.invoke({"topic": "the internet"})

# Asynchronous
result = await chain.ainvoke({"topic": "the internet"})

# Batch processing
results = chain.batch([{"topic": "the internet"}, {"topic": "AI"}])

# With configuration and tracing
result = chain.invoke(
    {"topic": "the internet"},
    config={"tags": ["poetry"], "metadata": {"version": "1.0"}}
)
```

**Core Methods:**
- `invoke(dict) -> PromptValue`: Synchronous formatting, returning a structured `PromptValue` (StringPromptValue or ChatPromptValue).
- `ainvoke(dict) -> PromptValue`: Asynchronous variant, enabling non-blocking formatting in async contexts.
- `batch(list[dict])`: Format multiple inputs simultaneously, useful for batch processing.
- `stream(dict)`: Streaming mode (rarely used for prompts; more common for downstream components like models).
- `get_input_schema()`: Returns a Pydantic model for input validation, used by agents and APIs.

**Return Types:**
- Prompts return `PromptValue` objects, not strings. `PromptValue` is an abstraction with two concrete types:
  - `StringPromptValue`: Contains a single string, used for PromptTemplate.
  - `ChatPromptValue`: Contains a list of `BaseMessage` objects with roles and content, used for ChatPromptTemplate.

## Prompt Composition

Prompts compose via the `+` operator, which concatenates messages and merges input variables. Composition creates a new immutable prompt instance without modifying originals.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# System context
system = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Your name is {bot_name}."),
])

# Chat history placeholder
history = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("chat_history"),
])

# User input
user_msg = ChatPromptTemplate.from_messages([
    ("human", "{user_input}"),
])

# Compose left-to-right
combined = system + history + user_msg

# Equivalent to:
# ChatPromptTemplate([
#     ("system", "You are a helpful assistant. Your name is {bot_name}."),
#     MessagesPlaceholder("chat_history"),
#     ("human", "{user_input}"),
# ])

result = combined.invoke({
    "bot_name": "Alice",
    "chat_history": [
        ("human", "Hi Alice"),
        ("ai", "Hello! How can I help?"),
    ],
    "user_input": "Tell me a joke."
})
```

**Composition Semantics:**
- **Message concatenation**: When combining `ChatPromptTemplate` instances, messages are concatenated in order.
- **Variable merging**: Input variables from all templates are collected into a single set. If both operands require the same variable, it appears once in the combined template.
- **Partial variable merging**: Partial variables from both operands are merged. If both define the same partial variable with different values, an error is raised.
- **Format compatibility**: All templates must use the same `template_format` (all f-string, all mustache, etc.). Mixing formats is not permitted.
- **Immutability**: The `+` operator returns a new instance; original templates are unchanged.

**Supported Operand Types:**
- `ChatPromptTemplate + ChatPromptTemplate` → concatenated messages
- `ChatPromptTemplate + BaseMessagePromptTemplate` (e.g., HumanMessagePromptTemplate) → appends single message
- `ChatPromptTemplate + BaseMessage` (e.g., SystemMessage instance) → appends raw message
- `ChatPromptTemplate + str` → converts to HumanMessagePromptTemplate and appends
- `ChatPromptTemplate + list/tuple` → converts to ChatPromptTemplate via `from_messages()` and concatenates

## Prompt Serialization

**Historical Note:** The old `save()` / `load_prompt_from_config()` API is deprecated. Use `dumpd()` / `loads()` from `langchain_core.load` instead.

### Serializing Prompts

Prompts can be serialized to dictionaries (JSON-compatible) using `dumpd()`:

```python
from langchain_core.load import dumpd
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
])

prompt_dict = dumpd(prompt)
# Returns: {"lc": 1, "type": "constructor", "id": [...], "kwargs": {...}}

# Save to JSON
import json
with open("prompt.json", "w") as f:
    json.dump(prompt_dict, f)
```

### Deserializing Prompts

Load prompts from dictionaries using `loads()`:

```python
from langchain_core.load import loads
import json

with open("prompt.json") as f:
    prompt_dict = json.load(f)

prompt = loads(prompt_dict)
# Returns the reconstructed PromptTemplate or ChatPromptTemplate
```

This enables version control, configuration management, and sharing of prompt definitions across teams.

## Integration with Agent Factory

Prompts are a core input to the **Agent Factory** (`create_agent`), providing the conversational context and system instructions for agent reasoning and tool use.

### Agent Factory Prompt Parameters

The `create_agent` function accepts prompts in two ways:

```python
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Method 1: Simple string (auto-wrapped as system message)
agent = create_agent(
    model="openai:gpt-4o",
    tools=[weather_tool],
    system_prompt="You are a helpful weather assistant."
)

# Method 2: Full ChatPromptTemplate (gives more control)
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful weather assistant. Your name is {agent_name}."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

agent = create_agent(
    model="openai:gpt-4o",
    tools=[weather_tool],
    prompt=template,
)
```

### Prompt Flow in Agents

The agent factory internally:
1. Compiles the system prompt with the model and tool bindings (via `model.bind_tools()`)
2. Manages message flow through the state machine, accumulating conversation history
3. Injects tool execution results as `ToolMessage` objects between model invocations
4. Passes the full message history to the model on each loop iteration

### Middleware Integration

Middleware can intercept and modify prompts before model invocation using the `wrap_model_call` hook. This enables:
- **Prompt optimization**: Rewrite prompts for brevity or clarity
- **Safety filtering**: Remove or redact sensitive instructions
- **Dynamic context injection**: Add retrieval results or real-time data to prompts
- **A/B testing**: Switch between prompt variants based on conditions

```python
from langchain.agents import create_agent

def optimize_prompt(messages, config):
    """Middleware that modifies the prompt."""
    # Filter messages, add context, etc.
    return messages

agent = create_agent(
    model="openai:gpt-4o",
    tools=[weather_tool],
    system_prompt="You are a helpful assistant.",
    middleware=[("wrap_model_call", optimize_prompt)],
)
```

See the **Agent Factory** documentation for full details on state machines, middleware stacks, and control flow.

## Security Considerations

### Template Injection

When constructing prompts from user input, use **partial variables** or **input variables** instead of string concatenation:

```python
# UNSAFE: Vulnerable to prompt injection
user_input = input("Enter text: ")
template = f"User said: {user_input}"  # Don't do this

# SAFE: Use variable substitution
from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate.from_template("User said: {user_input}")
output = prompt.format(user_input=user_input)
```

### Jinja2 Sandboxing

When using Jinja2 templates, LangChain applies `SandboxedEnvironment` by default to block access to dunder attributes (`__class__`, `__globals__`, etc.). This is a defense-in-depth measure, but:

- **Do not accept Jinja2 templates from untrusted sources.** Sandboxing is best-effort, not foolproof.
- Regular method calls and attribute access are still allowed (e.g., `obj.method()`, `obj.attr`).
- For applications requiring user-provided templates, **prefer f-string or mustache over Jinja2**, as they have simpler syntax with fewer escape paths.

```python
# Safe: f-string template from user, validated at construction time
from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate(
    template="Hello {name}",  # User-provided, but simple variable syntax
    input_variables=["name"],
)
```

## Lifecycle and State

Prompt templates are **immutable** in the functional sense: calling `format()` or `invoke()` does not mutate the template. Methods like `partial()` return new instances.

```python
original = PromptTemplate.from_template("Say {text}")
partial = original.partial(text="hello")  # Returns a NEW PromptTemplate

# original is unchanged
print(original.input_variables)  # ['text']
print(partial.input_variables)   # []
```

This immutability enables safe composition and caching in pipelines.

## Error Handling and Validation

Prompts validate configuration at construction time and inputs at format time.

### Construction-Time Validation

**Restricted Variable Names:**

Variable names `"stop"` are reserved internally and will raise a `ValueError`:

```python
from langchain_core.prompts import PromptTemplate

try:
    prompt = PromptTemplate.from_template("{stop}")  # Invalid
except ValueError:
    print("Cannot use 'stop' as a variable name")
```

**Overlapping Variables:**

If the same name appears in both `input_variables` and `partial_variables`, a `ValueError` is raised:

```python
try:
    prompt = PromptTemplate(
        template="{x}",
        input_variables=["x"],
        partial_variables={"x": "value"},  # Conflict
    )
except ValueError:
    print("Variable 'x' cannot be both required and partial")
```

**Few-Shot Template Constraints:**

`FewShotPromptTemplate` and `FewShotChatMessagePromptTemplate` require exactly one of `examples` or `example_selector`:

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

try:
    template = FewShotPromptTemplate(
        examples=[...],
        example_selector=selector,  # Error: both provided
        example_prompt=PromptTemplate.from_template("{input}"),
        suffix="...",
    )
except ValueError:
    print("Provide either 'examples' or 'example_selector', not both")
```

### Format-Time Validation

**Missing Variables:**

Calling `format()` or `invoke()` with missing required variables raises `KeyError`:

```python
prompt = PromptTemplate.from_template("{a} and {b}")
try:
    prompt.format(a="hello")  # Missing 'b'
except KeyError as e:
    print(f"Missing variable: {e}")  # "b"
```

**Extra Variables:**

Extra variables are silently ignored (not considered errors):

```python
prompt = PromptTemplate.from_template("{a}")
output = prompt.format(a="hello", b="ignored")  # 'b' is ignored
```

## Observability and Tracing

All prompts support LangChain's standard tracing and observability hooks via the Runnable interface:

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("human", "{input}"),
])

# Add metadata via with_config()
template_with_metadata = template.with_config({
    "run_name": "my_prompt",
    "metadata": {"version": "1.0", "owner": "team_a"},
    "tags": ["important", "production"],
})

# Metadata propagates to invoke() calls
result = template_with_metadata.invoke({"input": "hello"})
```

**Tracing Integration:**

- **LangSmith**: Metadata and tags appear in LangSmith traces, enabling performance analysis, debugging, and cost tracking.
- **Callbacks**: Prompts emit `on_llm_start` and `on_llm_end` events if callbacks are attached via config.
- **Run ID**: Each invocation receives a unique `run_id` for correlation with downstream model and tool calls.

This enables comprehensive observability across the entire chain: prompt → model → tools → output parser.
