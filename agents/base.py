"""
Base utilities shared across all agents.

Provides:
  - run_with_tools()       — runs a tool-calling agent via native OpenAI tools API
  - run_without_tools()    — runs a simple LLM call with no tools
  - parse_json_output()    — safely parses LLM JSON response
  - AgentOutput            — typed wrapper for agent results

When vLLM is launched with --enable-auto-tool-choice --tool-call-parser hermes,
the server handles tool call parsing natively. LangChain's ChatOpenAI.bind_tools()
sends tools as structured API parameters — no prompt engineering needed.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import get_llm, USE_NATIVE_TOOL_CALLING


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


# ── Native Tool Calling Prompt ────────────────────────────────────────────────
# Much simpler than the old structured chat prompt. The tools are passed
# as structured API parameters by ChatOpenAI.bind_tools(), not embedded
# in the prompt text. This saves ~500 tokens of context per agent call.

TOOL_CALLING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


def run_with_tools(
    system_prompt: str,
    human_message: str,
    tools: list,
    agent_name: str,
) -> AgentOutput:
    """
    Run a tool-calling agent using vLLM's native tool call support.

    When vLLM is started with --enable-auto-tool-choice, it handles
    tool call parsing at the server level. LangChain's bind_tools()
    sends tools as structured parameters in the OpenAI API request.

    This replaces the old structured-chat approach which required
    complex prompt engineering and custom output parsers.

    Set USE_NATIVE_TOOL_CALLING=false when vLLM is started without
    --enable-auto-tool-choice / --tool-call-parser (plain OpenAI-compatible server).
    """
    if not USE_NATIVE_TOOL_CALLING:
        return _run_with_tools_legacy(system_prompt, human_message, tools, agent_name)

    llm = get_llm()

    prompt = TOOL_CALLING_PROMPT.partial(system_prompt=system_prompt)

    try:
        agent = create_tool_calling_agent(llm, tools, prompt)
    except Exception:
        # Fallback: if the model doesn't advertise tool support,
        # use the legacy structured chat approach
        return _run_with_tools_legacy(system_prompt, human_message, tools, agent_name)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=8,
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


def _run_with_tools_legacy(
    system_prompt: str,
    human_message: str,
    tools: list,
    agent_name: str,
) -> AgentOutput:
    """
    Legacy fallback: structured-chat agent for models without native tool support.
    Uses prompt-based tool calling with a custom output parser.
    """
    from langchain.agents.structured_chat.output_parser import StructuredChatOutputParser
    from langchain.agents.format_scratchpad import format_log_to_str
    from langchain.tools.render import render_text_description_and_args
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.agents import AgentAction, AgentFinish
    from typing import Union

    class LenientStructuredChatOutputParser(StructuredChatOutputParser):
        """Extended parser that handles raw JSON final answers."""

        def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
            try:
                return super().parse(text)
            except Exception as original_error:
                action_match = re.search(
                    r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL
                )
                if action_match:
                    try:
                        response = json.loads(action_match.group(1).strip())
                        if isinstance(response, dict) and "action" not in response:
                            return AgentFinish(
                                {"output": json.dumps(response)}, text
                            )
                    except (json.JSONDecodeError, KeyError):
                        pass

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

    LEGACY_PROMPT = ChatPromptTemplate.from_messages([
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

    llm = get_llm()
    tool_strings = render_text_description_and_args(tools)
    tool_names = ", ".join([t.name for t in tools])

    prompt = LEGACY_PROMPT.partial(
        system_prompt=system_prompt,
        tools=tool_strings,
        tool_names=tool_names,
    )
    llm_with_stop = llm.bind(stop=["Observation"])

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(
                x["intermediate_steps"]
            ),
        )
        | prompt
        | llm_with_stop
        | LenientStructuredChatOutputParser()
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=8,
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