"""Research Agent Implementation.

This module implements a research agent that can perform iterative web searches
and synthesis to answer complex research questions.
"""

import os
from typing_extensions import Literal
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, START, END

from deep_research.state import ResearcherState, ResearcherOutputState
from deep_research.utils import tavily_search, get_today_str, think_tool
from deep_research.prompt import research_system_prompt, compress_research_system_prompt, compress_research_simple_human_message

load_dotenv()

# ===== CONFIGURATION =====

# Default to Google Gemini Flash-Lite
_DEFAULT_RESEARCH_MODEL = "google_genai:gemini-3.5-flash-lite"
_DEFAULT_COMPRESSION_MODEL = "google_genai:gemini-3.5-flash-lite"

tools = [tavily_search, think_tool]
tools_by_name = {tool.name: tool for tool in tools}

# ===== AGENT NODES =====

def llm_call(state: ResearcherState, config: RunnableConfig):
    """Analyze current state and decide on next actions."""
    model_name = config.get("configurable", {}).get("research_model", _DEFAULT_RESEARCH_MODEL)
    model = init_chat_model(model_name, temperature=0.7)
    model_with_tools = model.bind_tools(tools)
    return {
        "researcher_messages": [
            model_with_tools.invoke(
                [SystemMessage(content=research_system_prompt)] + state["researcher_messages"]
            )
        ]
    }

def tool_node(state: ResearcherState, config: RunnableConfig):
    """Execute all tool calls from the previous LLM response."""
    tool_calls = state["researcher_messages"][-1].tool_calls

    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observations.append(tool.invoke(tool_call["args"], config=config))

    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    iterations = state.get("tool_call_iterations", 0) + 1
    return {"researcher_messages": tool_outputs, "tool_call_iterations": iterations}

def compress_research(state: ResearcherState, config: RunnableConfig) -> dict:
    """Compress research findings into a concise summary."""
    model_name = config.get("configurable", {}).get("compression_model", _DEFAULT_COMPRESSION_MODEL)
    compress_model = init_chat_model(model_name, temperature=0.7)

    system_message = compress_research_system_prompt.format(date=get_today_str())
    messages = [SystemMessage(content=system_message)] + state.get("researcher_messages", []) + [HumanMessage(content=compress_research_simple_human_message)]
    response = compress_model.invoke(messages)

    # Extract clean text from response content
    if isinstance(response.content, list):
        compressed_text = "\n".join(
            p.get("text", str(p)) if isinstance(p, dict) else getattr(p, "text", str(p))
            for p in response.content
        )
    else:
        compressed_text = str(response.content)

    # Extract raw notes from tool and AI messages
    raw_notes = [
        str(m.content) for m in filter_messages(
            state["researcher_messages"], 
            include_types=["tool", "ai"]
        )
    ]

    return {
        "compressed_research": compressed_text,
        "raw_notes": ["\n".join(raw_notes)]
    }

# ===== ROUTING LOGIC =====

def should_continue(state: ResearcherState, config: RunnableConfig) -> Literal["tool_node", "compress_research"]:
    """Route to tool execution or compress based on tool calls and iteration limit."""
    last_message = state["researcher_messages"][-1]
    max_calls = config.get("configurable", {}).get("max_react_tool_calls", 10)
    iterations = state.get("tool_call_iterations", 0)

    if last_message.tool_calls and iterations < max_calls:
        return "tool_node"
    return "compress_research"

# ===== GRAPH CONSTRUCTION =====

# Build the agent workflow
agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# Add nodes to the graph
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_research", compress_research)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node", # Continue research loop
        "compress_research": "compress_research", # Provide final answer
    },
)
agent_builder.add_edge("tool_node", "llm_call") # Loop back for more research
agent_builder.add_edge("compress_research", END)

# Compile the agent
researcher_agent = agent_builder.compile()
