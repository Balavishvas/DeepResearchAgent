"""Scoping Agent Implementation.

This module determines if the user's request contains sufficient information
to proceed with research, clarifies ambiguities, and generates a structured research brief.
"""

import os
from typing_extensions import Literal
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from deep_research.prompt import (
    transform_messages_into_research_topic_prompt,
    clarify_with_user_instructions,
)
from deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ResearchQuestion,
)
from deep_research.utils import get_today_str

load_dotenv()

# ===== CONFIGURATION =====
_DEFAULT_SCOPING_MODEL = os.environ.get("SCOPING_MODEL", "gemini-3.5-flash-lite")

scoping_model = ChatGoogleGenerativeAI(
    model=_DEFAULT_SCOPING_MODEL,
    temperature=1.0,
    max_tokens=None,
    timeout=None,
)

# ===== AGENT NODES =====

def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief", "__end__"]]:
    """
    Determine if the user's request contains sufficient information to proceed with research.
    
    Uses structured output to make deterministic decisions and avoid hallucination.
    Routes to either research brief generation or ends with a clarification question.
    """
    structured_output_model = scoping_model.with_structured_output(ClarifyWithUser)

    response = structured_output_model.invoke([
        HumanMessage(
            content=clarify_with_user_instructions.format(
                messages=get_buffer_string(messages=state.get("messages", [])),
                date=get_today_str(),
            )
        )
    ])

    if response.need_clarification:
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=response.question)]},
        )
    else:
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]},
        )


def write_research_brief(state: AgentState) -> dict:
    """
    Transform the conversation history into a comprehensive research brief.
    
    Uses structured output to ensure the brief follows the required format
    and contains all necessary details for effective research.
    """
    structured_output_model = scoping_model.with_structured_output(ResearchQuestion)

    response = structured_output_model.invoke([
        HumanMessage(
            content=transform_messages_into_research_topic_prompt.format(
                messages=get_buffer_string(state.get("messages", [])),
                date=get_today_str(),
            )
        )
    ])

    return {
        "research_brief": response.research_brief,
        "supervisor_messages": [HumanMessage(content=f"{response.research_brief}.")],
    }


# ===== GRAPH CONSTRUCTION =====

deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", END)

scoping_agent = deep_researcher_builder.compile()
scope_research = scoping_agent
