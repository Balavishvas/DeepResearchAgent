"""
Full Multi-Agent Research System

This module integrates all components of the research system:
- User clarification and scoping
- Research brief generation  
- Multi-agent research coordination
- Final report generation

The system orchestrates the complete research workflow from initial user
input through final report delivery.
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.utils import get_today_str
from deep_research.prompt import final_report_generation_prompt
from deep_research.state import AgentState, AgentInputState
from deep_research.agents.scoping_agent import clarify_with_user, write_research_brief
from deep_research.agents.supervisor_agent import supervisor_agent

load_dotenv()

# ===== CONFIGURATION =====
_DEFAULT_FINAL_REPORT_MODEL = os.environ.get(
    "FINAL_REPORT_MODEL",
    "google_genai:gemini-3.5-flash-lite" if "GEMINI_API_KEY" in os.environ or "GOOGLE_API_KEY" in os.environ else "anthropic:claude-haiku-4-5-20251001"
)

async def final_report_generation(state: AgentState, config: RunnableConfig = None):
    """
    Final report generation node.
    
    Synthesizes all research findings into a comprehensive final report
    """
    config_dict = config.get("configurable", {}) if config else {}
    model_name = config_dict.get("final_report_model", _DEFAULT_FINAL_REPORT_MODEL)
    
    try:
        writer_model = init_chat_model(model_name, temperature=0.7)
    except Exception:
        writer_model = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            temperature=0.7,
            max_tokens=32000,
            timeout=None,
        )

    notes = state.get("notes", [])
    findings = "\n".join(notes) if notes else "No specific notes gathered."

    final_report_prompt = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str()
    )

    final_report = await writer_model.ainvoke([HumanMessage(content=final_report_prompt)])
    
    # Handle both string and list formats from Gemini / multimodal models
    if isinstance(final_report.content, list):
        report_text = "".join(
            item.get("text", "") if isinstance(item, dict) else getattr(item, "text", str(item))
            for item in final_report.content
        )
    else:
        report_text = str(final_report.content)

    return {
        "final_report": report_text, 
        "messages": [AIMessage(content=f"Here is the final report:\n\n{report_text}")],
    }

# ===== GRAPH CONSTRUCTION =====

deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add workflow nodes
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("supervisor_subgraph", supervisor_agent)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

# Add workflow edges
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", "supervisor_subgraph")
deep_researcher_builder.add_edge("supervisor_subgraph", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

# Checkpointer for persistent thread execution
checkpointer = InMemorySaver()
full_agent = deep_researcher_builder.compile(checkpointer=checkpointer)
agent = full_agent
