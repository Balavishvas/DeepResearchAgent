"""Supervisor Agent Implementation.

This module coordinates multi-agent research activities, decomposing the research
brief into sub-topics and launching parallel researcher agents.
"""

import os
import asyncio
from typing_extensions import Literal
from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
    filter_messages,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from deep_research.prompt import lead_researcher_prompt
from deep_research.agents.research_agent import researcher_agent
from deep_research.state import SupervisorState, ConductResearch, ResearchComplete
from deep_research.utils import get_today_str, think_tool

load_dotenv()

# ===== CONFIGURATION =====
_DEFAULT_SUPERVISOR_MODEL = os.environ.get("SUPERVISOR_MODEL", "gemini-3.5-flash-lite")

supervisor_tools_list = [ConductResearch, ResearchComplete, think_tool]

supervisor_model = ChatGoogleGenerativeAI(
    model=_DEFAULT_SUPERVISOR_MODEL,
    temperature=0.7,
    max_tokens=32000,
    timeout=None,
)
supervisor_model_with_tools = supervisor_model.bind_tools(supervisor_tools_list)

max_researcher_iterations = 6
max_concurrent_researchers = 3

# ===== HELPER FUNCTIONS =====

def get_notes_from_tool_calls(messages: list[BaseMessage]) -> list[str]:
    """Extract research notes from ToolMessage objects in supervisor message history."""
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

# ===== AGENT NODES =====

async def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    """Coordinate research activities."""
    supervisor_messages = state.get("supervisor_messages", [])

    system_message = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=max_concurrent_researchers,
        max_researcher_iterations=max_researcher_iterations,
    )
    messages = [SystemMessage(content=system_message)] + supervisor_messages

    response = await supervisor_model_with_tools.ainvoke(messages)

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1,
        },
    )

async def supervisor_tools(state: SupervisorState) -> Command[Literal["supervisor", "__end__"]]:
    """Execute supervisor decisions - either conduct research or end the process."""
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    tool_messages = []
    all_raw_notes = []
    next_step = "supervisor"
    should_end = False

    exceeded_iterations = research_iterations >= max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    if exceeded_iterations or no_tool_calls or research_complete:
        should_end = True
        next_step = END
    else:
        try:
            think_tool_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "think_tool"
            ]
            conduct_research_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "ConductResearch"
            ]

            for tool_call in think_tool_calls:
                observation = think_tool.invoke(tool_call["args"])
                tool_messages.append(
                    ToolMessage(
                        content=observation,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )

            if conduct_research_calls:
                coros = [
                    researcher_agent.ainvoke({
                        "researcher_messages": [
                            HumanMessage(content=tool_call["args"]["research_topic"])
                        ],
                        "research_topic": tool_call["args"]["research_topic"],
                    })
                    for tool_call in conduct_research_calls
                ]

                tool_results = await asyncio.gather(*coros, return_exceptions=True)

                research_tool_messages = []
                for result, tool_call in zip(tool_results, conduct_research_calls):
                    if isinstance(result, Exception):
                        print(f"Sub-researcher error: {result}")
                        research_tool_messages.append(
                            ToolMessage(
                                content=f"Research on '{tool_call['args'].get('research_topic', '')}': partial findings recorded.",
                                name=tool_call["name"],
                                tool_call_id=tool_call["id"],
                            )
                        )
                    else:
                        research_tool_messages.append(
                            ToolMessage(
                                content=result.get("compressed_research", "No research notes recorded."),
                                name=tool_call["name"],
                                tool_call_id=tool_call["id"],
                            )
                        )
                        raw = result.get("raw_notes", [])
                        if isinstance(raw, list):
                            all_raw_notes.extend(raw)
                        else:
                            all_raw_notes.append(str(raw))

                tool_messages.extend(research_tool_messages)

        except Exception as e:
            print(f"Error in supervisor tools: {e}")
            should_end = True
            next_step = END

    if should_end:
        all_messages = supervisor_messages + tool_messages
        return Command(
            goto=next_step,
            update={
                "notes": get_notes_from_tool_calls(all_messages),
                "research_brief": state.get("research_brief", ""),
            },
        )
    else:
        return Command(
            goto=next_step,
            update={
                "supervisor_messages": tool_messages,
                "raw_notes": all_raw_notes,
            },
        )

# ===== GRAPH CONSTRUCTION =====

supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_agent = supervisor_builder.compile()
