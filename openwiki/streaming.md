---
type: "Concept"
title: "Streaming: Tokens, Events, and Tool Outputs"
description: "How streaming works across LLM components and agents: token-by-token delivery via AIMessageChunk, event-level streaming with stream_events(), tool output deltas via ToolRuntime, and error handling during streaming."
tags: [streaming, token-streaming, event-streaming, llm-output, tool-output, chat-models, agents, callbacks, backpressure, astream, real-time-feedback]
sources:
  - id: openwiki-source-c9313cf42f0120d86b20245f
    resource: repo://libs/core/langchain_core/callbacks/base.py
  - id: openwiki-source-c7a2c3ef4ec61c3e28011205
    resource: repo://libs/core/langchain_core/callbacks/streaming_stdout.py
  - id: openwiki-source-5f8bc32563177d89fbab9b2f
    resource: repo://libs/core/langchain_core/language_models/chat_model_stream.py
  - id: openwiki-source-c52037e7b642f7ac5a7642a8
    resource: repo://libs/core/langchain_core/language_models/chat_models.py
  - id: openwiki-source-77dc1fb726463969f9d53658
    resource: repo://libs/core/langchain_core/messages/ai.py
  - id: openwiki-source-b32b84365d17276620c41ebc
    resource: repo://libs/core/langchain_core/messages/base.py
  - id: openwiki-source-8bb392f5dbc1fe7faaf52430
    resource: repo://libs/core/langchain_core/messages/human.py
  - id: openwiki-source-dad8cfeb38a829e03e165986
    resource: repo://libs/core/langchain_core/messages/system.py
  - id: openwiki-source-a1981e868973f6fd7f71e12e
    resource: repo://libs/core/langchain_core/runnables/base.py
  - id: openwiki-source-ea6411b24fd22a66368f2170
    resource: repo://libs/langchain_v1/langchain/tools/__init__.py
  - id: openwiki-source-ff9b926753e6cc6fe87acfe7
    resource: repo://libs/langchain_v1/tests/unit_tests/agents/test_agent_streaming.py
generated: { by: "openwiki/0.5.0", at: "2026-10-15T08:27:20.856Z" }
verified:
  - by: openwiki/0.5.0
    at: 2026-10-15T08:27:20.856Z
---

## Overview

**Streaming** is the mechanism by which LangChain delivers output incrementally during model and agent execution, rather than waiting for the entire response. This enables real-time feedback in web UIs, console displays, and observability tools, and forms the foundation for building responsive applications that do not block on model latency.

LangChain implements three primary streaming patterns:

1. **Token streaming** (`stream()` / `astream()`): Delivers individual tokens or content blocks as `AIMessageChunk` objects. Callbacks fire `on_llm_new_token` for each token, enabling real-time observation and UI updates.

2. **Event streaming** (`stream_events(version='v3')`): Yields structured protocol events (text-delta, tool_call_chunk, usage, etc.) with typed projections (`.text`, `.tool_calls`, `.usage`) that accumulate state. Implements pull-based backpressure to control producer pacing.

3. **Tool output streaming** (via `runtime.emit_output_delta()`): Tools can emit partial results mid-execution, visible in agent streams and tool call projections before the tool completes.

Streaming flows through chains—prompts, models, output parsers, and other runnables—preserving incremental output delivery at each stage. By composition, a chain automatically supports streaming if all its components do. This page documents the mechanics of all three streaming patterns, their control flow, integration with callbacks and agents, error handling, and memory/latency trade-offs.

## Synchronous Streaming: stream()

**Location**: `repo://libs/core/langchain_core/language_models/chat_models.py#L727-L856`

`BaseChatModel.stream()` is the primary synchronous streaming entry point. It yields `AIMessageChunk` objects as they are produced by the underlying model, with incremental content—a single token, a fragment of JSON, or a structured block update.

### Control Flow

1. **Check if streaming is implemented**: `_should_stream()` determines whether the model supports streaming. It checks:
   - Whether `_stream()` is implemented on the model (not inherited from base)
   - Whether streaming is explicitly disabled via `disable_streaming`, `stream=False`, or `streaming=False` on the model
   - Whether an explicit `stream=True` kwarg is passed
   - Whether a streaming callback handler is attached to the model
   
   If streaming is not supported, `stream()` falls back to `invoke()` and yields one complete result cast to `AIMessageChunk`.

2. **Initialize callbacks**: A `CallbackManager` is configured from the provided `RunnableConfig`, binding callbacks, tags, and metadata for tracing and observability.

3. **Fire on_chat_model_start**: The callback lifecycle begins with `on_chat_model_start`, signaling that LLM invocation is beginning. This event is fired before the first token is yielded.

4. **Acquire rate limit**: If a rate limiter is attached to the model, `stream()` acquires a permit before beginning, blocking until the rate limit allows.

5. **Iterate model chunks**: For each `ChatGenerationChunk` from the underlying `_stream()` implementation:
   - The chunk's message ID is set to a unique run ID (prefixed with `LC_ID_PREFIX`) if not already present, ensuring traceability.
   - Response metadata (model provider, latency, token usage, etc.) is computed and attached via `_gen_info_and_msg_metadata()`.
   - If the model's output version is "v1" (content-block structured format), content is transformed to content blocks and indexed.
   - **on_llm_new_token is fired** with the chunk's content and the full chunk object, allowing callbacks to observe or buffer each token.
   - The chunk message is cast to `AIMessageChunk` and yielded immediately to the caller.
   - Chunks are accumulated in memory for later aggregation.

6. **Yield final "last" chunk**: After the model finishes, if no explicit `chunk_position="last"` was set, an empty chunk with `chunk_position="last"` is yielded. This signals to parsers and consumers that the stream is complete and that `tool_call_chunks` should be finalized into complete `tool_calls`.

7. **Callback lifecycle closes**: If successful, `on_llm_end` fires with a merged `ChatGeneration` containing all chunks. If an exception occurs, `on_llm_error` fires with partial accumulation, allowing callbacks to observe failures before the exception is re-raised.

### Fallback Behavior

If the model does not implement streaming (checked via `_should_stream(async_api=False)`), `stream()` delegates to `invoke()` and yields a single result cast to `AIMessageChunk`. This ensures all models provide a consistent streaming interface, even if only non-streaming invoke is available.

## Asynchronous Streaming: astream()

**Location**: `repo://libs/core/langchain_core/language_models/chat_models.py#L858-L991`

`BaseChatModel.astream()` is the async variant of `stream()`, mirroring the synchronous logic but using async/await and `AsyncCallbackManager`.

**Key differences**:
- Uses `await` for callback events (`await run_manager.on_llm_new_token(...)`, `await run_manager.on_llm_end(...)`)
- Iterates via `async for chunk in self._astream(...)`
- Acquires rate limit via `await self.rate_limiter.aacquire(blocking=True)`

The async streaming protocol is identical to sync: check `_should_stream(async_api=True)`, initialize callbacks, yield chunks immediately as they arrive, fire callbacks per token, finalize tool call chunks on the "last" signal.

### Async/Await Patterns

Applications using `astream()` should consume the async iterator in a loop:

```python
async for chunk in model.astream(messages):
    # Process chunk immediately
    print(chunk.content, end="", flush=True)

# OR collect chunks for later processing
chunks = []
async for chunk in model.astream(messages):
    chunks.append(chunk)
final_message = sum(chunks)  # Merge via + operator
```

## Message Chunks: Types and Composition

**Location**: `repo://libs/core/langchain_core/messages/ai.py#L418-L536`

During streaming, models emit **chunk variants** of standard message types. All chunk types inherit from `BaseMessageChunk` and are designed to be **mergeable** via the `+` operator, enabling efficient aggregation of streaming output.

### Message Chunk Types

Each core message type has a corresponding chunk variant:

- **`AIMessageChunk`**: Emitted by chat models during streaming. Contains:
  - `content`: String or list of content blocks (delta for that step).
  - `tool_call_chunks`: List of `ToolCallChunk` objects (incomplete tool calls being streamed).
  - `chunk_position`: Optional `"last"` sentinel signaling stream completion.
  - `response_metadata`: Model metadata (latency, provider, usage counts, etc.).

- **`HumanMessageChunk`**: Represents incremental user input. Used in streaming scenarios where user messages arrive in chunks (e.g., real-time voice transcription or chunked uploads).

- **`SystemMessageChunk`**: Represents incremental system prompt updates. Less common than AIMessageChunk but available for consistency.

- **`ToolMessageChunk`**: Represents incremental tool results. Supports streaming tool outputs that arrive in chunks (e.g., file downloads, streaming API responses).

All chunk types support the `+` operator for merging, making them composable at any level of a streaming pipeline.

### AIMessageChunk Structure

`AIMessageChunk` is the message type yielded during streaming. Unlike `AIMessage`, it represents a **partial, incremental update** to a conversation message and supports merging via the `+` operator.

**Fields**:
- **content**: String or list of content blocks (when `output_version="v1"`). During streaming, each chunk contains only the new token(s) or delta for that step. Content accumulates across chunks: text chunks concatenate, JSON chunks may append partial objects or arrays.
- **tool_call_chunks**: List of `ToolCallChunk` objects (incomplete tool calls being streamed). These are progressively updated as the model produces call ID, function name, and argument JSON. Arguments are accumulated and parsed incrementally via `parse_partial_json()`.
- **chunk_position**: Optional sentinel; when set to `"last"`, indicates the final chunk in the stream, triggering finalization of tool calls and reasoning blocks. When this chunk is aggregated, `tool_call_chunks` are parsed into complete `tool_calls` and `invalid_tool_calls` via the `init_tool_calls()` validator.
- **response_metadata**: Model-specific metadata (latency, model_provider, usage counters, finish reason, etc.) attached by the streaming handler. Metadata is merged across chunks, with usage counts summed.

### Merging Chunks via the + Operator

Streaming chunks accumulate via the `+` operator (implemented in `add_ai_message_chunks()`), which:

1. **Merges content**: Text content is concatenated; structured content blocks are merged according to block type.
2. **Concatenates tool_call arguments**: Arguments from tool_call_chunks are appended progressively, enabling incremental JSON parsing.
3. **Combines metadata**: Response metadata and usage counts are merged; for duplicated keys, later values override (except usage, which is summed).
4. **Preserves chunk_position**: If any chunk in the merge has `chunk_position="last"`, the result marks position as "last", triggering tool call finalization.
5. **Selects best ID**: The chunk ID is chosen by rank: provider-assigned (non-`LC_*` prefixed) > `LC_run_*` > `lc_*` auto IDs.

A complete `AIMessage` with finalized `tool_calls` (not chunks) is reconstructed when chunks are merged or when the "last" signal is received:

```python
# Accumulate chunks
chunks = []
async for chunk in model.astream(messages):
    chunks.append(chunk)
    
# Merge all chunks into one message
final_message = chunks[0]
for chunk in chunks[1:]:
    final_message = final_message + chunk
    
# tool_calls are now complete ToolCall objects, not ToolCallChunk
for tool_call in final_message.tool_calls:
    print(tool_call["name"], tool_call["args"])
```

### Chunk Composition Flow

The following diagram shows how chunks flow through a streaming pipeline and merge into complete messages:

```mermaid
sequenceDiagram
    participant Model
    participant StreamIterator as stream()<br/> Iterator
    participant Callback as on_llm_new_token<br/> Callback
    participant Aggregator as Chunk<br/> Accumulator
    participant Merger as Chunk<br/> Merge (+ op)
    participant Complete as Final<br/> AIMessage

    Model->>StreamIterator: yield AIMessageChunk<br/> content="Hello"
    StreamIterator->>Callback: fire on_llm_new_token<br/> token="Hello"
    StreamIterator->>Aggregator: buffer chunk
    StreamIterator-->>Model: pull next
    
    Model->>StreamIterator: yield AIMessageChunk<br/> content=" world"
    StreamIterator->>Callback: fire on_llm_new_token<br/> token=" world"
    StreamIterator->>Aggregator: buffer chunk
    StreamIterator-->>Model: pull next
    
    Model->>StreamIterator: yield AIMessageChunk<br/> chunk_position="last"
    StreamIterator->>Callback: fire on_llm_new_token<br/> token=""
    StreamIterator->>Aggregator: buffer chunk
    
    Note over Aggregator: All chunks buffered
    Aggregator->>Merger: merge chunks:<br/> chunk1 + chunk2 + chunk_last
    Merger->>Complete: single AIMessage<br/> with finalized tool_calls
    Callback->>Callback: fire on_llm_end<br/> with merged result
```

The diagram shows how token-by-token chunks are yielded immediately, callbacks are fired for real-time observation, and final aggregation occurs only at the end.

## Token-Level Streaming in Action

Streaming operates at the **token level**, not the message level. Here's how tokens flow through a typical streaming sequence:

### Text Token Streaming

```python
# Model generates: "The capital of France is Paris"

from langchain_core.callbacks import StreamingStdOutCallbackHandler

messages = [HumanMessage("What is the capital of France?")]
callback = StreamingStdOutCallbackHandler()

for chunk in model.stream(messages, config=RunnableConfig(callbacks=[callback])):
    # Each iteration receives one chunk
    # Chunks might be: "The", " capital", " of", " France", " is", " Paris"
    # callback.on_llm_new_token fires for each, writing to stdout immediately
    
    # chunk.content is the token string
    # chunk.tool_call_chunks is empty (no tools called)
    # chunk.response_metadata has model latency, provider, etc.
    pass
```

The `StreamingStdOutCallbackHandler` writes each token to stdout immediately, producing real-time text output without waiting for the full response.

### Tool Call Streaming

Tool call arguments stream across multiple chunks:

```python
# Model generates a tool call with JSON args: 
# get_weather(city="Paris", unit="celsius")

chunks = []
async for chunk in model.astream(messages):
    chunks.append(chunk)
    
    # Each chunk might contain partial tool call data:
    # Chunk 1: tool_call_chunks=[{"id": "tc1", "name": "get_weather", "args": ""}]
    # Chunk 2: tool_call_chunks=[{"id": "tc1", "name": "get_weather", "args": "{\"ci"}]
    # Chunk 3: tool_call_chunks=[{"id": "tc1", "name": "get_weather", "args": "...ty\":\"Paris\",\"unit\":"}]
    # Chunk 4: tool_call_chunks=[{"id": "tc1", "name": "get_weather", "args": "...\"celsius\"}"]
    # Chunk 5 (last): chunk_position="last" triggers finalization
    
    if chunk.tool_call_chunks:
        for tc in chunk.tool_call_chunks:
            # parse_partial_json handles incomplete JSON gracefully
            args = parse_partial_json(tc.get("args", ""))
            print(f"Tool: {tc['name']}, partial args: {args}")

# Merge to get final message with complete tool_calls
final_message = sum(chunks[1:], chunks[0])
for tool_call in final_message.tool_calls:
    print(f"Final: {tool_call['name']}({tool_call['args']})")
```

At each step, callbacks and custom handlers can observe partial tool call construction, enabling real-time visibility into tool invocation even before arguments are complete.

## Callback Integration: on_llm_new_token

**Location**: `repo://libs/core/langchain_core/callbacks/base.py#L65-L88`

The `on_llm_new_token` callback fires for each token or chunk during streaming, enabling real-time observation and logging.

### Callback Signature

```python
def on_llm_new_token(
    self,
    token: str | list[str | dict[str, Any]],
    *,
    chunk: GenerationChunk | ChatGenerationChunk | None = None,
    run_id: UUID,
    parent_run_id: UUID | None = None,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> Any:
```

- **token**: The string token or list of content blocks (when output_version="v1"). For text streaming, this is a single word or subword; for structured output, this is a list of content block dicts with `type`, `text`, `reasoning`, `tool_call_chunk`, etc.
- **chunk**: The full `ChatGenerationChunk` carrying metadata, message ID, response metadata, and tool_call_chunks. This allows callbacks to inspect the complete chunk structure, not just the token.
- **run_id**: Unique identifier for this streaming run, used for tracing and correlation with parent operations.
- **parent_run_id**: ID of the parent run (chain or agent) that invoked this model.
- **tags**: Inheritable tags from the calling context, useful for filtering or routing callbacks.

### Built-in: StreamingStdOutCallbackHandler

```python
from langchain_core.callbacks import StreamingStdOutCallbackHandler

callback = StreamingStdOutCallbackHandler()

# Callbacks are passed via RunnableConfig
for chunk in model.stream(
    messages,
    config=RunnableConfig(callbacks=[callback])
):
    pass  # callback prints each token to stdout
```

The `StreamingStdOutCallbackHandler` implements `on_llm_new_token` to write tokens to `sys.stdout` immediately, making streaming output visible in real-time without buffering.

### Custom Streaming Callbacks

Create custom callbacks by subclassing `BaseCallbackHandler`:

```python
from langchain_core.callbacks import BaseCallbackHandler
from typing import Any

class MyStreamingCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        # Send token to WebSocket, log to database, etc.
        websocket.send_json({"token": token})
    
    def on_llm_end(self, response, **kwargs: Any) -> None:
        # Final response complete
        websocket.send_json({"status": "complete"})
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        # Error occurred during streaming
        websocket.send_json({"error": str(error)})
```

### LangSmith Integration

LangSmith automatically captures streaming via callbacks. When you use `stream()` with LangSmith callbacks attached:

1. **run_id** from `on_llm_new_token` is used to trace the streaming operation.
2. Each token fires `on_llm_new_token`, allowing LangSmith to record token-level granularity.
3. `on_llm_end` provides the final merged result, allowing LangSmith to record total latency and token counts.
4. If streaming fails, `on_llm_error` records the failure with partial results.

LangSmith projects these traces in the UI as token-by-token progress, enabling real-time observability of model streaming behavior.

## Error Handling in Streaming

**Location**: `repo://libs/core/langchain_core/language_models/chat_models.py#L833-L856`

When an exception occurs during streaming, LangChain ensures error visibility to callbacks while preserving partial results for inspection.

### Exception Flow

1. **During streaming**: If an error occurs after `on_chat_model_start` has fired but before the stream completes:
   - Any chunks already yielded are accumulated in memory.
   - The exception is caught by the streaming handler.
   - `on_llm_error` fires with the accumulated partial results as a `ChatGeneration`, allowing callbacks to inspect what was generated before failure.
   - The exception is re-raised to the caller, propagating the error up the call stack.

2. **Callback observation**: Callbacks can distinguish between partial and complete results:
   - `on_llm_end` fires only on success with complete results.
   - `on_llm_error` fires on failure with whatever was accumulated before the error.
   - Both callbacks receive run_id for tracing, allowing error correlation with upstream operations.

### Error Handling Example

```python
class ErrorTrackingCallback(BaseCallbackHandler):
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        # Capture partial results even if streaming fails
        run_id = kwargs.get("run_id")
        print(f"Streaming error on run {run_id}: {error}")
        # Partial results are available if the error occurred mid-stream
        # Example: model connection dropped after 50 tokens were sent

for chunk in model.stream(
    messages,
    config=RunnableConfig(callbacks=[ErrorTrackingCallback()])
):
    # If an error occurs, the exception is raised here
    # but callbacks have already seen the partial result
    print(chunk.content, end="", flush=True)
```

This pattern ensures that partial results are not lost and can be logged or presented to users even when streaming fails.

## Streaming Through Chains

Streaming flows through chains composed of runnables (prompts, models, parsers). The streaming protocol is implemented at each stage via the `stream()` and `transform()` methods on `Runnable`.

### Default Behavior

**Location**: `repo://libs/core/langchain_core/runnables/base.py#L1194-L1235`

By default, `Runnable.stream()` yields one full output from `invoke()`. Subclasses that support streaming override `stream()` or `transform()` to yield chunks. The `transform()` method is the core streaming interface: it accepts an iterator of inputs and yields an iterator of outputs, enabling stateful transformations.

### Streaming through RunnableSequence

**Location**: `repo://libs/core/langchain_core/runnables/base.py#L3075-L3320`

`RunnableSequence` (a chain created with the `|` operator) automatically supports streaming if:

1. **All upstream components implement transform**: The `transform()` method maps streaming input to streaming output, enabling end-to-end streaming without buffering.
2. **The last component produces chunks**: Output parsers and models implement `transform()` to yield partial results.

If any component does not implement `transform()`, streaming begins only after that component completes (blocking point). Multiple blocking components create multiple buffering points, but the final output still streams from the last component if it supports streaming.

**Important**: `RunnableLambda` does not implement `transform()` by default, so it acts as a blocking component. For custom logic with streaming, subclass `Runnable` and override `transform()`.

### Streaming Example: Model → Parser

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI()
parser = StrOutputParser()
chain = model | parser

# stream yields parser outputs incrementally as tokens arrive
for chunk in chain.stream("What is 2+2?"):
    print(chunk, end="", flush=True)
```

When `model.stream()` yields chunks, the parser's `transform()` (inherited from `BaseTransformOutputParser`) consumes each chunk and yields its transformation. Text parsers (like `StrOutputParser`) extract text from `AIMessageChunk` and yield strings directly; JSON parsers yield partial JSON objects as they become parseable via `parse_partial_json()`.

### Streaming Mechanics: _transform_stream_with_config

**Location**: `repo://libs/core/langchain_core/runnables/base.py#L2502-L2599`

The `_transform_stream_with_config()` helper manages streaming with callbacks. It:

1. **Tees the input iterator** so the first element can be inspected for tracing without consuming it.
2. **Fires on_chain_start** before the transformer begins, signaling the start of a streaming chain operation.
3. **Invokes the transformer function** with the remaining input iterator and child callbacks.
4. **Yields chunks immediately** as the transformer produces them, enabling responsive streaming.
5. **Accumulates outputs** for the on_chain_end callback, optionally merging chunks via `+` if supported.
6. **Fires on_chain_end or on_chain_error** at completion, providing final merged output or exception context.

This mechanism ensures streaming callbacks fire for each chunk and that parent run managers know when a chain's streaming is complete.

## Streaming via stream_events: ChatModelStream

**Location**: `repo://libs/core/langchain_core/language_models/chat_model_stream.py`

For advanced use cases requiring detailed event granularity, `BaseChatModel.stream_events(version="v3")` returns a `ChatModelStream` object that exposes **typed projection properties** (`.text`, `.tool_calls`, `.usage`, `.reasoning`, `.output`) which accumulate protocol events as they arrive.

### Structured Event Streaming

Unlike token streaming (`stream()`), which yields tokens, `stream_events()` yields **protocol events**—structured objects representing model state changes:

- **text-delta**: Incremental text generation
- **reasoning-delta**: Incremental reasoning/thinking content (when supported)
- **tool_call_chunk**: Partial tool call with accumulated arguments
- **usage**: Token usage update (input, output, cached, etc.)

### Pull-Based Backpressure

The `ChatModelStream` and its projections (`.text`, `.tool_calls`, etc.) implement **pull-based backpressure** via the `SyncProjection` and `AsyncProjection` classes. When a consumer reads from a projection and catches up to the buffer:

1. The projection calls `_request_more()` to pull additional events from the producer (the model/graph).
2. The producer resumes and generates the next batch of events.
3. The projection buffers events and yields them to the consumer.

This backpressure mechanism prevents unbounded memory growth: the producer only generates events as the consumer requests them. Unlike callback-driven streaming (which delivers all tokens as fast as the model produces them), pull-based streaming allows the consumer to set the pace.

**Example: Consuming with backpressure**

```python
# Pull events on demand; producer waits if no consumer is pulling
for event in model.stream_events(messages, version="v3"):
    if should_stop_early():
        break  # Producer stops; no buffered events accumulate
    process_event(event)

# Or consume a specific projection with type safety
stream = model.stream_events(messages, version="v3")
for text_delta in stream.text:  # Only text events
    print(text_delta)
```

### Backpressure and Cancellation

**Token streaming** (`stream()` / `astream()`): Uses callback-driven delivery. The producer (model) generates tokens as fast as it can; the consumer does not pace the producer. Callbacks fire immediately for each token. If the consumer loop breaks early, the stream iterator is disposed but the model may continue generating tokens internally until completion or the connection closes.

**Event streaming** (`stream_events(version='v3')`): Uses pull-based backpressure. The producer waits until the consumer reads from a projection. If the consumer loop breaks early:
- The projection stops requesting events.
- The producer naturally pauses (no unbounded buffering).
- Resources are released when the projection goes out of scope or is explicitly closed.

When a consumer calls `break` or exits a loop consuming a projection:
- The projection is no longer subscribed to the mux.
- Other projections (if active) continue receiving events.
- The producer may continue generating events if other consumers are active.
- Once all consumers are done, the producer completes and cleanup occurs.

This design allows multiple independent consumers (e.g., `.text`, `.tool_calls`) to observe the same stream without interference, with each consumer able to control its own pacing via backpressure.

## Streaming in Agent Execution

**Location**: `repo://libs/langchain_v1/tests/unit_tests/agents/test_agent_streaming.py`, `repo://libs/langchain_v1/langchain/agents/factory.py`

Agents created via `create_agent()` support streaming execution via `stream_events(version="v3")` on the returned agent graph. This enables real-time visibility into the agent lifecycle at event granularity rather than token granularity.

### Agent Streaming Lifecycle

When an agent runs with `stream_events(version="v3")`, the following sequence occurs:

1. **Initial state**: Agent graph begins with user input messages.
2. **Model invocation**: The language model is called. `on_llm_start` fires. Tokens stream via `on_llm_new_token` callbacks and are visible in stream events.
3. **Tool calls emitted**: Model produces tool call chunks. These accumulate in the `run.tool_calls` projection, with each tool call accessible incrementally.
4. **Tool execution**: Each tool is invoked. `on_tool_start` fires. If the tool calls `runtime.emit_output_delta()`, those deltas appear in `tool_call.output_deltas`.
5. **Tool results**: `on_tool_end` fires. Tool results are wrapped in `ToolMessage` and added to the agent state.
6. **Loop decision**: The agent decides whether to continue looping (more tool calls) or exit (no more tools).
7. **Hierarchical tools**: If a tool invokes a nested agent via `create_agent()`, its tool calls are visible via `run.subgraphs` with their own `tool_calls` projection.
8. **Final output**: Agent completes. The final message list is accessible via `run.messages` and `run.output` projections.

### Real-Time Visibility

During this entire lifecycle, consumers can observe events as they fire:

1. **Real-time visibility into tool calls**: The `run.tool_calls` projection accumulates tool calls as the model generates them. Each tool call is a `ToolCallStream` object with `tool_name`, `tool_call_id`, `args`, `completed`, `error`, and `output_deltas`.

2. **Output deltas before completion**: Tools can stream output via `runtime.emit_output_delta()` before the tool completes. The `tool_call.output_deltas` projection makes these deltas visible immediately, enabling UIs to show tool progress without waiting for the tool to finish.

3. **Messages projection**: The `run.messages` projection contains the full message history as it accumulates. Each model invocation adds an `AIMessage`, each tool result adds a `ToolMessage`.

4. **Hierarchical observability**: When a tool delegates to a nested agent, `run.subgraphs` exposes subgraph handles, each with their own `tool_calls` projection. This allows observing both outer and inner agent operations independently.

### Tool Output Streaming

Tools can emit output deltas mid-execution via the `ToolRuntime` injected parameter. `ToolRuntime.emit_output_delta()` allows tools to send partial results asynchronously, independent of tool completion:

```python
from langchain_core.tools import tool
from langchain.tools import ToolRuntime

@tool
def long_running_task(query: str, runtime: ToolRuntime) -> str:
    """A task that streams output as it runs."""
    for chunk in process_incrementally(query):
        # Emit partial output immediately; visible in streams
        runtime.emit_output_delta(chunk)
    
    # Return final result (separate from streamed deltas)
    return compute_final_result(query)

@tool
async def async_streaming_tool(query: str, runtime: ToolRuntime) -> str:
    """Async tool emitting output deltas."""
    async for result in async_process(query):
        runtime.emit_output_delta(result)
    return final_result(query)

# When streaming the agent, output deltas appear in tool_call projections:
run = agent.stream_events({"messages": [HumanMessage("go")]}, version="v3")

for tool_call in run.tool_calls:
    print(f"Tool: {tool_call.tool_name}")
    # output_deltas contains all emitted chunks before tool completes
    for delta in tool_call.output_deltas:
        print(f"  Delta: {delta}")
    print(f"  Final result: {tool_call.output}")  # Merged after completion
```

The agent stream captures these deltas in real-time, making them visible to observers before the tool completes. Deltas are accumulated in the `output_deltas` projection per tool call, separate from the final `output` result.

### Agentic Loop Streaming

The agent execution loop itself can be streamed:

```python
from langchain.agents import create_agent

agent = create_agent(model, tools)

# Stream events from the agent graph
for event in agent.stream_events(
    {"messages": [HumanMessage("Question")]}, 
    version="v3"
):
    # Event types: "on_llm_start", "on_llm_new_token", "on_tool_start", 
    # "on_tool_end", "on_chain_end", custom events, etc.
    if event["type"] == "on_llm_new_token":
        print(f"Token: {event['content']}")
    elif event["type"] == "on_tool_start":
        print(f"Tool: {event['tool']}")
    elif event["type"] == "on_tool_end":
        print(f"Tool result: {event['output']}")
```

### Tool Call Streaming with Projections

The most convenient way to stream agent execution is via projections:

```python
run = agent.stream_events({"messages": [HumanMessage("go")]}, version="v3")

# Stream tool calls as they're generated
for tool_call in run.tool_calls:
    print(f"Tool call: {tool_call.tool_name}")
    print(f"Args: {tool_call.args}")
    
    # Stream tool output deltas if the tool emits them
    for output_delta in tool_call.output_deltas:
        print(f"Output: {output_delta}")

# Also available: run.messages, run.text, run.usage, run.output
```

### Hierarchical Agent Streaming

When agents invoke sub-agents via tools:

```python
# Outer agent calls a tool that delegates to an inner agent
for event in outer_agent.stream_events({...}, version="v3"):
    if event_type == "subgraph":
        # A sub-agent ran; access its projections
        subgraph = event["subgraph"]
        for inner_tool_call in subgraph.tool_calls:
            print(f"Inner tool: {inner_tool_call.tool_name}")
```

Streaming hierarchical agents allows observing both the outer loop (which tools the outer agent calls) and inner loops (which tools nested agents call), providing full visibility into complex agent compositions.

## Memory and Latency Trade-offs: stream() vs invoke()

### invoke()

- **Latency**: Waits for the entire model response before returning. Introduces latency equal to the full model generation time.
- **Memory**: No intermediate storage required; only the final message is held in memory.
- **Responsiveness**: Blocks the calling thread/coroutine until complete. Users see no output until the response is fully generated.
- **Use case**: Batch processing, when a complete response is needed upfront before proceeding to the next step.

### stream()

- **Latency**: Yields the first token as soon as available; responsive to user. Time to first token (TTFT) is minimized.
- **Memory**: Requires buffering of accumulated chunks if the caller collects them. However, because chunks are yielded immediately, the caller can process and discard each chunk without holding the entire response.
- **Responsiveness**: Non-blocking; enables progressive display. Users see output appearing in real-time.
- **Use case**: Web UIs, console applications, user-facing interactions where real-time feedback improves UX.

### Streaming Does Not Add Latency

In practice, streaming does not add significant latency compared to invoke; the model produces tokens at the same rate. The difference is **when tokens are delivered to the caller**. Stream delivery is preferable for interactive applications because users see output appearing in real-time rather than a blank screen until the full response is ready.

### Backpressure and Memory Implications

When streaming with `stream()`:

- **Callback-driven delivery**: Tokens are yielded as fast as the model produces them. If the caller is slow to consume, tokens accumulate in the accumulator list within `stream()` until the loop ends or yields.
- **No unbounded growth**: The chunk accumulator is only used for the final `on_llm_end` callback; chunks are yielded immediately before accumulating. Thus, memory overhead is proportional to the response size, not model speed.
- **Consumer pacing**: Slow consumers (e.g., writing to disk) do not create backpressure; they simply process tokens as yielded.

When streaming with `stream_events()` (v3):

- **Pull-based backpressure**: The producer (model/graph) only generates events as the consumer requests them via the projection iterator. This naturally paces the producer to the consumer.
- **Bounded buffering**: The projection buffers events only until the consumer reads them. A slow consumer will naturally slow the producer, preventing unbounded memory growth.
- **Multiple independent consumers**: Multiple `for` loops over different projections (e.g., `.text` and `.tool_calls`) can replay all events from the buffer, supporting diverse consumption patterns without re-running the model.

## Best Practices for Streaming

1. **Flush output immediately**: When displaying streaming output in web or terminal, flush buffers after each chunk to ensure immediate visibility.

2. **Handle partial JSON carefully**: JSON parsers should use `parse_partial_json()` to extract complete structures from partial JSON as tokens arrive, rather than waiting for the full response.

3. **Merge chunks for final use**: If you need the complete response, collect chunks and merge them via `+`:
   ```python
   chunks = [chunk for chunk in model.stream(messages)]
   final = chunks[0]
   for chunk in chunks[1:]:
       final = final + chunk
   ```

4. **Use callbacks for side effects**: Implement `on_llm_new_token` for logging, metrics, and webhooks rather than processing each yielded chunk in the loop. Callbacks decouple application logic from streaming concerns.

5. **Respect backpressure**: When using `stream_events()`, let the consumer pace the producer. Don't artificially speed up event generation.

6. **Disable streaming selectively**: For long-running operations or when you need predictable latency, use `invoke()` instead of `stream()`, or pass `stream=False` to override the default.

7. **Test both sync and async paths**: Streaming behavior may differ between `stream()` and `astream()` depending on model implementation and callback executors. Test both for your use case.

8. **Observe tool calls incrementally**: When tools are involved, use `on_llm_new_token` with full `chunk` inspection to see tool call arguments accumulating, enabling early detection of tool invocation intent before arguments are complete.

9. **Use `stream_events()` for complex workflows**: For agents, hierarchical tools, and workflows with multiple streaming sources, prefer `stream_events(version="v3")` for precise event-level control and pull-based backpressure.
