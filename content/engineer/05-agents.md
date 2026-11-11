# 5. Agents

## Core idea

An **agent** is a program that lets a large language model (LLM) do a task in more than one step.
The model writes what it wants to do. A normal program does it and gives the result back to the model.
This loop runs again and again until the model writes the answer.

## The example

The [toy example](../toy-example.md) has no tools, so this page uses its own small task.

| Item | Value |
|---|---|
| Task | "How many legs do 3 cats have?" |
| Tool | `calculator`: one arithmetic operation on two numbers |
| Model | A scripted model: a list of 2 fixed replies. There is no API call. |
| Correct answer | "3 cats have 12 legs." |

A **tool** is a function that the program can run for the model.
The scripted model gives the same replies on each run, so you can check each number on this page.

## 1. Agent fundamentals

### Setup

One turn of the loop has three parts:

- **Thought**: text in which the model decides what to do next.
- **Action**: a request to run one tool with given arguments. It is also called a **tool call**.
- **Observation**: the result of the tool, given back to the model as text.

The model reads the definition of each tool. A definition has a name, a description and a **JSON schema** (a description of the permitted arguments and their types).

```json
{
  "name": "calculator",
  "description": "Do one arithmetic operation on two numbers.",
  "input_schema": {
    "type": "object",
    "properties": {
      "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "required": ["operation", "a", "b"]
  }
}
```

<figure class="diagram" data-src="agents/loop.svg">
<figcaption>The loop for the cat task. The model writes text. Your code runs the tool.</figcaption>
</figure>

### Steps

1. **Send the task and the tool definitions to the model.** The input has 2 messages: the system instruction and the user question.

    **Why:** the model can only ask for a tool that it can read about. The description is the only manual that the model has.

2. **The model writes a thought and an action.** This is the tool-call message:

    ```json
    {"role": "assistant",
     "thought": "One cat has 4 legs. I need 3 x 4.",
     "tool_call": {"id": "call_1", "name": "calculator",
                   "arguments": {"operation": "multiply", "a": 3, "b": 4}}}
    ```

    **Why:** the model cannot run code. It can only write text. A fixed JSON shape lets a program read that text without a guess.

3. **The loop checks the call, then runs the tool.** The name `calculator` exists. The 3 arguments agree with the schema. The function gives 3 × 4 = 12.

    **Why:** a model can write a wrong tool name or forget an argument. The check finds the error before your code runs with bad input.

4. **The loop adds the observation to the messages.** This is the tool-result message:

    ```json
    {"role": "tool", "tool_call_id": "call_1", "content": "12"}
    ```

    **Why:** the model has no memory between two calls. It sees the result only if the result is in its next input. The id connects the result to the call.

5. **Call the model again with the full history.** The input now has 4 messages. The model writes a thought and the answer "3 cats have 12 legs." There is no tool call, so the loop stops. Total: 2 model calls and 1 tool call.

    **Why:** the model decides when the task is done. The loop needs only one stop rule: a reply without a tool call is the answer. A step limit (here 5) stops a model that never answers.

6. **Calculate the reliability of a chain of steps.** Assume that each step succeeds with a probability $p$, and that the steps are independent. Then a chain of $n$ steps succeeds with:

    $$P(\text{all } n \text{ steps succeed}) = p^n$$

    | Steps $n$ | 1 | 5 | 10 | 20 |
    |---|---|---|---|---|
    | $0.95^n$ | 0.950 | 0.774 | 0.599 | 0.358 |

    **Why:** an agent has no spare step. One wrong tool call can make each later step wrong. A step that is correct 95% of the time looks good, but a chain of 20 such steps fails about 2 times out of 3.

<figure class="plot">
<img src="/figures/agents/reliability.svg" alt="Line chart of p to the power n for p = 0.99, 0.95 and 0.90. For p = 0.95 the values are 0.950, 0.774, 0.599 and 0.358 at 1, 5, 10 and 20 steps.">
<figcaption>The success of a chain falls fast with its length. The marked points are the 4 values for p = 0.95.</figcaption>
</figure>

<div class="key-point">

An agent is a loop around a model: thought, action, observation. The model only writes text. Your code runs each tool, so your code controls what the agent can do.

</div>

### From toy to real

- This loop is known as **ReAct** (reason + act). Real model APIs return the tool call in a dedicated field. The field names change with the vendor, but the content is the same JSON schema and the same arguments.
- **Errors are observations.** In a second scripted run, the model asks for a tool named `multiplier`. The loop answers `Error: unknown tool 'multiplier'`. The model reads the error and calls `calculator`. That run needs 3 model calls, not 2.
- **Length limits.** With $p = 0.95$, the longest chain that still succeeds 1 time out of 2 has 13 steps. For 90% success on 20 steps, each step must succeed with $0.9^{1/20} = 0.9947$.
- **Checks help.** If the loop detects each failure and tries the step one more time, a step succeeds with $1 - 0.05^2 = 0.9975$. A chain of 20 steps then succeeds with 0.951. The formula $p^n$ is a model: real steps are not fully independent.
- **Cost.** Each model call reads the full history again: 2 messages, then 4. A long task sends a longer input at each step.

## 2. Agent protocols

### Setup

Without a standard, each application needs special code for each tool. Assume 3 applications and 4 tools: that is 3 × 4 = 12 pieces of connection code. A **protocol** is an agreed message format. With a protocol, each side writes its code one time: 3 + 4 = 7 pieces.

<figure class="diagram" data-src="agents/protocols.svg">
<figcaption>MCP connects an application to tools and data. A2A connects an agent to another agent.</figcaption>
</figure>

### Steps

1. **MCP: the application asks a server for its tools.** The Model Context Protocol (MCP) has three roles. The **host** is the AI application. The **MCP server** is a program that offers tools and data. The host makes one **MCP client** for each server. The client sends `tools/list` and gets a name, a description and an `inputSchema` for each tool.

    **Why:** the application does not need to know the tools when you write it. It finds them at run time.

2. **MCP: the client calls the tool.** The messages use JSON-RPC 2.0 (a simple standard for "call this method with these parameters"). For the cat task:

    ```json
    {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
     "params": {"name": "calculator",
                "arguments": {"operation": "multiply", "a": 3, "b": 4}}}
    ```

    ```json
    {"jsonrpc": "2.0", "id": 1,
     "result": {"resultType": "complete",
                "content": [{"type": "text", "text": "12"}]}}
    ```

    A real request also has a `_meta` field with the protocol version. It is left out here.

    **Why:** each tool has the same message shape. One client works with all servers.

3. **MCP: a server offers three types of items.**

    | Item | What it is | Example |
    |---|---|---|
    | Tool | A function that the model can call | `calculator` |
    | Resource | Data that the application can read | A file, a database schema |
    | Prompt | A text template that the user can select | "Review this code" |

    **Why:** a tool can change something, so it needs more control than data that is only read.

4. **A2A: an agent gives work to another agent.** In the Agent2Agent protocol (A2A), a **client agent** sends a message to a **remote agent**. The remote agent publishes an **Agent Card**: a JSON file at `/.well-known/agent-card.json` with its name, address, skills and authentication rules. The client calls `SendMessage`:

    ```json
    {"message": {"role": "ROLE_USER", "messageId": "msg-1",
                 "parts": [{"text": "How many legs do 3 cats have?"}]}}
    ```

    The reply is a message, or a **task**: a unit of work with an id and a state (for example working, then completed). The results of a task are its **artifacts**.

    **Why:** the remote agent is a black box. The client does not see its tools or its prompt. It only sends work and gets results.

<div class="key-point">

MCP is for "agent to tool". A2A is for "agent to agent". They do different jobs, and one system can use both.

</div>

### From toy to real

- MCP has two transports. **Stdio** is for a local server that runs as a process on your machine. **Streamable HTTP** is for a remote server with many clients.
- A2A is a Linux Foundation project. It has three bindings: JSON-RPC, gRPC and HTTP+JSON. For long tasks the client can poll, open a stream, or get push notifications.
- A tool result is text from outside your application. It can contain instructions that you did not write. See [Securing LLMs](08-securing.md).

Real-library sketch (the tests do not run this code): the calculator as an MCP server with the Python SDK.

```python
from mcp.server import MCPServer

mcp = MCPServer("calculator")

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

## 3. Vendor frameworks

### Setup

The loop of sub-part 1 is the same in each agent. An agent **framework** is a library that contains this loop, so you write only the tools and the instructions. Each large model vendor has one.

### Steps

1. **Write the tool as a normal function.** The framework reads the function name, the argument types and the docstring. It builds the JSON schema from them.

    **Why:** a schema that you write by hand can become different from the code. A generated schema cannot.

2. **Give the instructions and the tools to an agent object, then start a run.** The framework does steps 1 to 5 of sub-part 1: it calls the model, checks the tool call, runs the tool, adds the result and stops.

    **Why:** the loop code is always the same. A tested library has fewer errors than a new loop in each project.

3. **Set the controls.** Examples: a step limit, a list of permitted tools, an approval step before a risky tool, and a trace of each step.

    **Why:** the reliability table shows that long chains fail. You need the trace to see which step failed, and the limits to stop a bad run.

| | OpenAI Agents SDK | Google ADK | Claude Agent SDK |
|---|---|---|---|
| Languages | Python, TypeScript | Python, TypeScript, Go, Java, Kotlin | Python, TypeScript |
| Main parts | Agents, handoffs (give the task to another agent), guardrails, sessions | LLM agents and workflow agents (sequential, parallel, loop) | The agent loop and the built-in tools of Claude Code (files, commands, web search) |
| Tools | Python functions, MCP servers | Functions, MCP, OpenAPI | Built-in tools, MCP servers |
| Extras | Tracing | Evaluation tools, A2A support, deployment on Google Cloud | Hooks, subagents, permissions, sessions |

<div class="key-point">

A vendor framework is the same loop with the controls already built. It is the shortest path if you use one model vendor. The cost is that your agent code depends on that vendor.

</div>

### From toy to real

These libraries change often. Read the current documentation before you copy an example. The data in the table comes from the documentation pages in **Sources**.

## 4. Other frameworks

### Setup

Other frameworks do not belong to a model vendor. They differ in the question that they answer: how to draw the workflow, how to connect data, or how to divide work between agents.

### Steps

1. **LangGraph: write the workflow as a graph.** A **graph** has nodes (steps) and edges (the permitted next steps). The cat task has 2 nodes, `model` and `tool`. The edge `model → tool` is used when there is a tool call. The shared state is the list of messages.

    **Why:** a graph makes the control flow visible. The framework can save the state after each node, so a run can stop for a human approval and continue later.

2. **LlamaIndex: connect the agent to your documents.** The search in your documents becomes a tool. The agent decides when to search and what to ask. See [Retrieval augmented generation](03-rag.md).

    **Why:** many tasks are questions about private data. The model did not see this data in training.

3. **CrewAI and AutoGen: divide the work between agents.** Each agent has a role, for example researcher and writer. Count the steps. If 3 agents each do 5 steps, the chain has 15 steps: $0.95^{15} = 0.463$.

    **Why:** an agent with one role has a shorter prompt and fewer tools, so each step can be more reliable. But more agents mean more steps and more messages. Measure before you add an agent.

| Framework | Main idea | Good for |
|---|---|---|
| LangGraph | Graph of nodes with a shared state, saved at checkpoints | Workflows that you must control, pause and continue |
| LlamaIndex | Agents that use your data (RAG) | Questions about documents |
| CrewAI | A team of agents with roles ("crew"), in a controlled "flow" | Work that divides into roles |
| AutoGen | Agents that talk to each other | Research prototypes. The README says that the project is archived. Microsoft names Microsoft Agent Framework as the replacement. |

<div class="key-point">

All these frameworks run the same thought, action, observation loop. Select one for its control flow and its tooling, not for the count of its features.

</div>

### From toy to real

- Your own loop (sub-part 1) is a valid option. Use a framework when you need what it adds: checkpoints, data connectors or roles.
- Examine the status of a project before you depend on it. The AutoGen project shows why: its README says that it is archived and not maintained.

## Minimal code

This is the loop of sub-part 1. `model` is any function that reads the messages and returns JSON text.

```python
import json

def run_agent(model, task, tools, max_steps=5):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    for step in range(max_steps):
        reply = json.loads(model(messages))             # thought + tool call, or thought + answer
        messages.append({"role": "assistant", **reply})
        if "answer" in reply:
            return reply["answer"]                      # "3 cats have 12 legs."
        call = reply["tool_call"]
        observation = call_tool(tools, call)            # checks the schema, runs the tool: "12"
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": observation})
    return None                                         # the step limit stopped the loop
```

## Comparison

| Level | What you get | Use it when | Risk |
|---|---|---|---|
| Your own loop | Full control, about 15 lines | The task is small, or you want to learn | You write each control yourself |
| Protocol (MCP, A2A) | Tools and agents that work with many applications | You share tools between applications or teams | Tool results are untrusted text |
| Vendor framework | The loop, tracing and controls for one vendor | You use one model vendor | Dependence on that vendor |
| Other framework | Graphs, data connectors, teams of agents | You need a special control flow or many models | More layers to debug |

## Checks and common misconceptions

- **"The model runs the tool."** It does not. The model writes a request, and your code runs the function. Check: your loop must refuse a tool name or an argument that is not in the schema.
- **"More agents make the system more reliable."** Each added step multiplies the success by $p$. Check: count the steps of your task, measure $p$ on real runs, and calculate $p^n$ before you build.
- **"Each task needs an agent."** If you know the steps before the run, write them as fixed code (a **workflow**). Use an agent only when the next step depends on the last result.

## Sources

This page follows section 5 of "The LLM Engineer" in the [LLM Course](https://github.com/mlabonne/llm-course) by Maxime Labonne (Apache-2.0). The text and the example are new.

- [Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction), Hugging Face.
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), Yao et al., 2022.
- [Model Context Protocol](https://modelcontextprotocol.io/) and its [architecture overview](https://modelcontextprotocol.io/docs/learn/architecture) (roles, primitives, transports, `tools/call` messages); [Build an MCP server](https://modelcontextprotocol.io/docs/develop/build-server) (Python sketch).
- [Agent2Agent protocol](https://a2a-protocol.org/): [key concepts](https://a2a-protocol.org/latest/topics/key-concepts/) and [specification](https://a2a-protocol.org/latest/specification/) (Agent Card, `SendMessage`, tasks, bindings).
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/), [Google ADK](https://google.github.io/adk-docs/), [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview).
- [LangGraph](https://www.langchain.com/langgraph) and its [overview](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/), [LlamaIndex agents](https://docs.llamaindex.ai/en/stable/use_cases/agents/), [CrewAI](https://docs.crewai.com/introduction), [AutoGen](https://github.com/microsoft/autogen) (archive notice in the README).
