"""Numbers and plots for content/engineer/05-agents.md.

Run: uv run python -m examples.engineer.agents
"""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass

from examples.plotstyle import save_figure

PAGE = "agents"

TASK = "How many legs do 3 cats have?"
SYSTEM = "Answer the question. Use the calculator tool for each calculation."


# --- The tool: a JSON schema that the model reads, and a function that the program runs -----

CALCULATOR_TOOL = {
    "name": "calculator",
    "description": "Do one arithmetic operation on two numbers.",
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["operation", "a", "b"],
    },
}


def calculator(operation: str, a: float, b: float) -> float:
    """The real tool. The model never runs this code; the agent loop does."""
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        return a / b
    raise ValueError(f"unknown operation '{operation}'")


TOOLS: dict[str, tuple[dict, Callable[..., float]]] = {"calculator": (CALCULATOR_TOOL, calculator)}

JSON_TYPES = {"string": str, "number": (int, float)}


def check_arguments(schema: dict, arguments: dict) -> str | None:
    """Check `arguments` against a flat JSON schema. Return the first problem, or None."""
    for key in schema["required"]:
        if key not in arguments:
            return f"missing argument '{key}'"
    for key, value in arguments.items():
        if key not in schema["properties"]:
            return f"unknown argument '{key}'"
        rule = schema["properties"][key]
        if isinstance(value, bool) or not isinstance(value, JSON_TYPES[rule["type"]]):
            return f"argument '{key}' must be a {rule['type']}"
        if "enum" in rule and value not in rule["enum"]:
            return f"argument '{key}' must be one of {rule['enum']}"
    return None


def call_tool(tools: dict, call: dict) -> str:
    """Run one tool call. An error becomes text that the model can read and correct."""
    if call["name"] not in tools:
        return f"Error: unknown tool '{call['name']}'"
    definition, function = tools[call["name"]]
    problem = check_arguments(definition["input_schema"], call["arguments"])
    if problem:
        return f"Error: {problem}"
    try:
        result = function(**call["arguments"])
    except ZeroDivisionError:
        return "Error: division by zero"
    return str(int(result)) if float(result).is_integer() else str(result)


# --- The model: a list of fixed replies (no API call) ---------------------------------------

class ScriptedModel:
    """A fake model. Call number i returns reply number i, as JSON text."""

    def __init__(self, replies: list[dict]):
        self.replies = [json.dumps(reply) for reply in replies]
        self.context_sizes: list[int] = []  # number of messages that each call received

    def __call__(self, messages: list[dict]) -> str:
        self.context_sizes.append(len(messages))
        return self.replies[len(self.context_sizes) - 1]


def multiply_call(call_id: str) -> dict:
    return {"id": call_id, "name": "calculator", "arguments": {"operation": "multiply", "a": 3, "b": 4}}


FINAL_REPLY = {"thought": "The tool result is 12. I can answer.", "answer": "3 cats have 12 legs."}

# The run on the page: one tool call, then the answer.
CAT_REPLIES = [
    {"thought": "One cat has 4 legs. I need 3 x 4.", "tool_call": multiply_call("call_1")},
    FINAL_REPLY,
]

# A run with one wrong tool name. The error message lets the model correct the call.
RECOVERY_REPLIES = [
    {"thought": "One cat has 4 legs. I need 3 x 4.",
     "tool_call": {"id": "call_1", "name": "multiplier", "arguments": {"a": 3, "b": 4}}},
    {"thought": "That tool does not exist. I use the calculator.", "tool_call": multiply_call("call_2")},
    FINAL_REPLY,
]


# --- The agent loop --------------------------------------------------------------------------

@dataclass
class AgentResult:
    answer: str | None  # None when the loop stopped at the step limit
    trace: list[tuple[str, str]]  # (kind, text), kind = thought / action / observation / answer
    messages: list[dict]
    model_calls: int


def run_agent(model: Callable[[list[dict]], str], task: str, tools: dict, max_steps: int = 5) -> AgentResult:
    """Thought -> action -> observation, until the model gives an answer or the step limit."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    trace: list[tuple[str, str]] = []
    for step in range(1, max_steps + 1):
        reply = json.loads(model(messages))
        messages.append({"role": "assistant", **reply})
        trace.append(("thought", reply["thought"]))
        if "answer" in reply:
            trace.append(("answer", reply["answer"]))
            return AgentResult(reply["answer"], trace, messages, step)
        call = reply["tool_call"]
        trace.append(("action", f"{call['name']}({json.dumps(call['arguments'])})"))
        observation = call_tool(tools, call)
        trace.append(("observation", observation))
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": observation})
    return AgentResult(None, trace, messages, max_steps)


# --- The same tool call in the Model Context Protocol (JSON-RPC 2.0) ------------------------

def mcp_messages(call: dict, observation: str) -> tuple[dict, dict]:
    """A `tools/call` request and its result. The `_meta` field of each request is left out."""
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": call["name"], "arguments": call["arguments"]}}
    response = {"jsonrpc": "2.0", "id": 1,
                "result": {"resultType": "complete", "content": [{"type": "text", "text": observation}]}}
    return request, response


def integrations(apps: int, tools: int) -> dict[str, int]:
    """Pieces of connection code to write: one for each pair, or one for each side of a protocol."""
    return {"custom": apps * tools, "protocol": apps + tools}


# --- Reliability of a chain of steps ----------------------------------------------------------

def chain_success(p: float, n: int) -> float:
    """Probability that n independent steps all succeed, when one step succeeds with p."""
    return p ** n


def longest_chain(p: float, target: float) -> int:
    """Largest number of steps for which the chain still succeeds with probability >= target."""
    return math.floor(math.log(target) / math.log(p))


def step_success_needed(target: float, n: int) -> float:
    """Per-step success probability that gives `target` for a chain of n steps."""
    return target ** (1 / n)


def with_one_retry(p: float) -> float:
    """Per-step success when each failure is detected and the step runs one more time."""
    return 1 - (1 - p) ** 2


# --- Plot --------------------------------------------------------------------------------------

def make_figures() -> None:
    steps = list(range(1, 31))
    marked = [1, 5, 10, 20]

    def draw(ax, theme):
        for i, p in enumerate([0.99, 0.95, 0.90]):
            ax.plot(steps, [chain_success(p, n) for n in steps], color=theme["series"][i],
                    label=f"p = {p:.2f} for each step")
        ax.scatter(marked, [chain_success(0.95, n) for n in marked], color=theme["series"][1], zorder=3)
        for n in marked:
            ax.annotate(f"{chain_success(0.95, n):.3f}", (n, chain_success(0.95, n)),
                        textcoords="offset points", xytext=(5, 6), fontsize=8, color=theme["ink2"])
        ax.set_xlabel("Number of steps in the chain (n)")
        ax.set_ylabel("Probability that all steps succeed")
        ax.set_xlim(0, 30.5)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left")

    save_figure(draw, PAGE, "reliability")


if __name__ == "__main__":
    print("tool definition:", json.dumps(CALCULATOR_TOOL, indent=2))
    model = ScriptedModel(CAT_REPLIES)
    result = run_agent(model, TASK, TOOLS)
    for kind, text in result.trace:
        print(f"{kind:>12}: {text}")
    for message in result.messages:
        print(json.dumps(message))
    print("model calls:", result.model_calls, "| messages seen by each call:", model.context_sizes)
    request, response = mcp_messages(CAT_REPLIES[0]["tool_call"], "12")
    print("MCP request:", json.dumps(request))
    print("MCP result:", json.dumps(response))

    recovery = run_agent(ScriptedModel(RECOVERY_REPLIES), TASK, TOOLS)
    print("recovery trace:", recovery.trace, "| model calls:", recovery.model_calls)

    print("3 apps and 4 tools:", integrations(3, 4))
    for n in (1, 5, 10, 15, 20):
        print(f"0.95^{n} = {chain_success(0.95, n):.3f}")
    print("longest chain with p=0.95 and success >= 0.5:", longest_chain(0.95, 0.5))
    print("per-step p needed for 0.9 over 20 steps:", round(step_success_needed(0.9, 20), 4))
    print("p with one retry:", round(with_one_retry(0.95), 4),
          "-> 20 steps:", round(chain_success(with_one_retry(0.95), 20), 3))
    make_figures()
