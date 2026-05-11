"""
Base utilities shared across all agents.

Provides:
  - run_with_tools()       — runs a structured-chat agent with tool access
  - run_without_tools()    — runs a simple LLM call with no tools
  - parse_json_output()    — safely parses LLM JSON response
  - AgentOutput            — typed wrapper for agent results
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Union

from langchain.agents import AgentExecutor
from langchain.agents.structured_chat.output_parser import StructuredChatOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.tools.render import render_text_description_and_args
from langchain_core.runnables import RunnablePassthrough
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import get_llm


@dataclass
class AgentOutput:
    """Standardised wrapper for what every agent returns."""
    agent_name: str
    success: bool
    data: dict = field(default_factory=dict)
    raw_response: str = ""
    error: str = ""
    tool_calls_made: int = 0


def parse_json_output(text: str, agent_name: str) -> dict:
    """
    Robustly extract a JSON object from an LLM response.

    LLMs sometimes wrap JSON in markdown fences (```json ... ```) or
    add preamble text before the object. This handles all common cases.
    """
    # Strip markdown fences
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first { ... } block
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        candidate = text[brace_start:brace_end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Give up — return a safe default with the raw text attached
    return {
        "_parse_error": True,
        "_raw": text[:500],
        "_agent": agent_name,
    }


# ── Custom Output Parser ──────────────────────────────────────────────────────
# DeepSeek and similar models often output their final answer as a raw JSON
# code block without the {"action": "Final Answer", "action_input": "..."}
# wrapper. The default StructuredChatOutputParser fails on these because it
# tries to read response["action"] and gets a KeyError.
#
# This parser handles that case by detecting JSON without an "action" key
# and treating it as a final answer.

class LenientStructuredChatOutputParser(StructuredChatOutputParser):
    """Extended parser that handles raw JSON final answers."""

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        try:
            return super().parse(text)
        except Exception as original_error:
            # Look for a JSON code block that might be a final answer
            action_match = re.search(
                r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL
            )
            if action_match:
                try:
                    response = json.loads(action_match.group(1).strip())
                    if isinstance(response, dict) and "action" not in response:
                        # It's a final-answer JSON without the action wrapper
                        return AgentFinish(
                            {"output": json.dumps(response)}, text
                        )
                except (json.JSONDecodeError, KeyError):
                    pass

            # Try to find any JSON object in the text
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start != -1 and brace_end != -1:
                candidate = text[brace_start : brace_end + 1]
                try:
                    response = json.loads(candidate)
                    if isinstance(response, dict) and "action" not in response:
                        return AgentFinish({"output": candidate}, text)
                except (json.JSONDecodeError, KeyError):
                    pass

            raise original_error


# ── Structured Chat Prompt ────────────────────────────────────────────────────

STRUCTURED_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Respond to the human as helpfully and accurately as possible.

{system_prompt}

You have access to the following tools:

{tools}

Use this JSON format to call a tool:

```
{{{{
  "action": $TOOL_NAME,
  "action_input": $INPUT_STRING
}}}}
```

The only values that should be in the "action" field are: {tool_names}

IMPORTANT: The "action_input" value must be a simple string, not a JSON object.
For example: "action_input": "onMessage"  (correct)
NOT: "action_input": {{{{"symbol": "onMessage"}}}}  (wrong)

Follow this format strictly:

Thought: think about what to do
Action:
```
{{{{"action": "tool_name", "action_input": "input_string"}}}}
```
Observation: the tool result appears here
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: your final JSON response here

CRITICAL RULES:
1. You MUST call tools and wait for real Observations. Do NOT hallucinate results.
2. After gathering tool results, prefix your answer with "Final Answer:" on its own line.
3. Do NOT repeat the same tool call you already made.
4. If a tool says you hit the call limit, IMMEDIATELY give your Final Answer."""),
    ("human", "{input}\n\n{agent_scratchpad}\n\n(reminder: use the format above)"),
])


def _build_agent(llm, tools, prompt):
    """Build a structured chat agent with the lenient output parser."""
    tool_strings = render_text_description_and_args(tools)
    tool_names = ", ".join([t.name for t in tools])

    prompt_with_tools = prompt.partial(
        tools=tool_strings, tool_names=tool_names
    )
    llm_with_stop = llm.bind(stop=["Observation"])

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(
                x["intermediate_steps"]
            ),
        )
        | prompt_with_tools
        | llm_with_stop
        | LenientStructuredChatOutputParser()
    )
    return agent


def run_with_tools(
    system_prompt: str,
    human_message: str,
    tools: list,
    agent_name: str,
) -> AgentOutput:
    """
    Run a structured-chat agent with JSON-formatted tool calls.

    Uses a custom agent chain with LenientStructuredChatOutputParser
    that handles DeepSeek's tendency to output raw JSON final answers
    without the action wrapper.

    Used by agents that need to call GitHub tools (Dependency Mapper,
    Change Simulator, Test Gap Agent).
    """
    llm = get_llm()

    prompt = STRUCTURED_CHAT_PROMPT.partial(system_prompt=system_prompt)
    agent = _build_agent(llm, tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=8,          # slightly higher to allow recovery
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    try:
        result = executor.invoke({"input": human_message})
        output_text = result.get("output", "")
        tool_calls = len(result.get("intermediate_steps", []))
        parsed = parse_json_output(output_text, agent_name)
        return AgentOutput(
            agent_name=agent_name,
            success=not parsed.get("_parse_error", False),
            data=parsed,
            raw_response=output_text,
            tool_calls_made=tool_calls,
        )
    except Exception as e:
        return AgentOutput(
            agent_name=agent_name,
            success=False,
            error=str(e),
        )


def run_without_tools(
    system_prompt: str,
    human_message: str,
    agent_name: str,
) -> AgentOutput:
    """
    Run a simple LLM call with no tools.

    Used by Risk Evaluator and Critic — they only reason over text,
    no GitHub API calls needed.
    """
    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message),
    ]

    try:
        response = llm.invoke(messages)
        output_text = response.content
        parsed = parse_json_output(output_text, agent_name)
        return AgentOutput(
            agent_name=agent_name,
            success=not parsed.get("_parse_error", False),
            data=parsed,
            raw_response=output_text,
        )
    except Exception as e:
        return AgentOutput(
            agent_name=agent_name,
            success=False,
            error=str(e),
        )