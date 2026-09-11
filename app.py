"""
Deep Research Agent - Academic & Enterprise Streamlit Application
Autonomous Multi-Agent Research System powered by LangGraph, Google Gemini, and Tavily.
"""

import os
import time
import json
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
import nest_asyncio
from langchain_core.messages import HumanMessage, AIMessage

# Enable nested event loops for Streamlit async operations
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Import the LangGraph agent pipeline
from deep_research.agents.full_agent import full_agent

# ==============================================================================
# PAGE CONFIGURATION & METADATA
# ==============================================================================
st.set_page_config(
    page_title="Deep Research Agent | Autonomous Multi-Agent AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Persistent History File
HISTORY_FILE = Path(__file__).resolve().parent / "data_history.json"


def load_history():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(history_list):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_list, f, indent=2)
    except Exception as e:
        st.error(f"Error saving history: {e}")


# ==============================================================================
# STYLING (Professional Enterprise & Academic Theme)
# ==============================================================================
CUSTOM_CSS = """
<style>
/* Modern typography and font smoothing */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #1e293b;
}

/* App Header Card */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
    padding: 24px 30px;
    border-radius: 16px;
    color: #ffffff;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15), 0 8px 10px -6px rgba(15, 23, 42, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.main-header h1 {
    font-size: 26px !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    margin: 0 0 6px 0 !important;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 12px;
}

.main-header p {
    font-size: 14px;
    color: #94a3b8;
    margin: 0;
    font-weight: 400;
}

/* Status Badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.badge-success {
    background-color: #dcfce7;
    color: #166534;
    border: 1px solid #bbf7d0;
}

.badge-warning {
    background-color: #fef3c7;
    color: #92400e;
    border: 1px solid #fde68a;
}

.badge-info {
    background-color: #e0f2fe;
    color: #0369a1;
    border: 1px solid #bae6fd;
}

/* Agent Step Cards */
.agent-step-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: all 0.2s ease;
}

.agent-step-card:hover {
    border-color: #cbd5e1;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.07);
}

.agent-step-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-weight: 600;
    font-size: 14px;
    color: #0f172a;
    margin-bottom: 8px;
}

/* Final Report Container */
.report-container {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 16px;
    padding: 32px 36px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
    line-height: 1.7;
}

.report-container h1, .report-container h2, .report-container h3 {
    color: #0f172a;
    font-weight: 700;
    letter-spacing: -0.01em;
}

.report-container h1 {
    font-size: 24px;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 12px;
    margin-top: 10px;
}

.report-container h2 {
    font-size: 19px;
    margin-top: 28px;
    color: #1e293b;
}

.report-container h3 {
    font-size: 16px;
    margin-top: 20px;
    color: #334155;
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background-color: #f8fafc;
    border-right: 1px solid #e2e8f0;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Metric mini card */
.stat-box {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
}

.stat-val {
    font-size: 20px;
    font-weight: 700;
    color: #0f172a;
}

.stat-label {
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 2px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "research_results" not in st.session_state:
    st.session_state.research_results = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "clarification_question" not in st.session_state:
    st.session_state.clarification_question = None
if "current_brief" not in st.session_state:
    st.session_state.current_brief = None
if "current_notes" not in st.session_state:
    st.session_state.current_notes = []


# ==============================================================================
# SIDEBAR CONFIGURATION
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    st.caption("Customize autonomous multi-agent parameters")

    # 1. Models Section
    with st.expander("🔧 Models Configuration", expanded=False):
        report_model = st.selectbox(
            "Final Report Model",
            options=["gemini-3.5-flash-lite", "gemini-2.5-flash"],
            index=0,
            help="High-capacity generative model responsible for synthesizing the comprehensive report.",
        )

        supervisor_model = st.selectbox(
            "Supervisor & Scoping Model",
            options=["gemini-3.5-flash-lite", "gemini-2.5-flash"],
            index=0,
            help="Orchestration model that scopes the topic, formulates research briefs, and directs researchers.",
        )

    # 2. Search & Research Parameters
    with st.expander("🔍 Search & Research Parameters", expanded=True):
        allow_clarification = st.checkbox(
            "Allow Clarification Questions",
            value=True,
            help="When checked, the scoping agent asks clarifying questions if user query lacks required specifics.",
        )

        max_concurrent_researchers = st.slider(
            "Max Concurrent Researchers",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
            help="Number of parallel researcher agents exploring independent sub-hypotheses simultaneously.",
        )

        supervisor_max_iterations = st.slider(
            "Supervisor Max Iterations",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
            help="Maximum orchestration rounds the supervisor agent can execute before final report synthesis.",
        )

        max_tool_calls = st.slider(
            "Max Tool Calls / Researcher",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Maximum search and reasoning tool calls each parallel researcher can make per sub-topic.",
        )

    st.markdown("---")

    # 3. Environment & API Health Check
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    tavily_key = os.environ.get("TAVILY_API_KEY")

    st.markdown("#### 🔑 Service Status")
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        if gemini_key:
            st.markdown("<span class='badge badge-success'>● Gemini API Active</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge badge-warning'>▲ Gemini Missing</span>", unsafe_allow_html=True)
    with col_k2:
        if tavily_key:
            st.markdown("<span class='badge badge-success'>● Tavily Active</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge badge-warning'>▲ Tavily Missing</span>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Reset / New Session Button
    if st.button("🗑️ Start New Research", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.research_results = None
        st.session_state.is_running = False
        st.session_state.clarification_question = None
        st.session_state.current_brief = None
        st.session_state.current_notes = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='font-size: 11px; color: #94a3b8; text-align: center;'>"
        "Deep Research Agent v1.0<br>Built with LangGraph & Google Gemini"
        "</div>",
        unsafe_allow_html=True,
    )


# ==============================================================================
# MAIN VIEW: HEADER & TABS
# ==============================================================================
st.markdown(
    """
    <div class="main-header">
        <h1>🔬 Deep Research Agent</h1>
        <p>Autonomous Multi-Agent Literature, Fact-Checking & Web Intelligence Pipeline</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_research, tab_architecture, tab_history = st.tabs([
    "💬 Autonomous Research", 
    "📊 Multi-Agent Pipeline Architecture", 
    "📜 Session History"
])


# ==============================================================================
# ASYNC RESEARCH EXECUTOR
# ==============================================================================
async def execute_research_pipeline(query_text: str, thread_id: str, is_clarification_response: bool = False):
    """Run the multi-agent LangGraph pipeline with step-by-step UI updates."""
    thread_config = {
        "configurable": {
            "thread_id": thread_id,
            "recursion_limit": 60,
            "research_model": f"google_genai:{supervisor_model}",
            "compression_model": f"google_genai:{supervisor_model}",
            "final_report_model": f"google_genai:{report_model}",
            "max_concurrent_researchers": max_concurrent_researchers,
            "max_researcher_iterations": supervisor_max_iterations,
            "max_react_tool_calls": max_tool_calls,
            "allow_clarification": allow_clarification,
        }
    }

    input_message = HumanMessage(content=query_text)
    
    agent_output = {
        "status": "in_progress",
        "brief": "",
        "notes": [],
        "final_report": "",
        "clarification_needed": False,
        "clarification_question": "",
    }

    progress_placeholder = st.empty()
    steps_container = st.container()

    with steps_container:
        step1_box = st.empty()
        step2_box = st.empty()
        step3_box = st.empty()
        step4_box = st.empty()

    step1_box.markdown(
        """
        <div class="agent-step-card">
            <div class="agent-step-header">
                <span>🎯 1. Scoping Agent</span>
                <span class="badge badge-info">Analyzing requirements...</span>
            </div>
            <div style="font-size: 13px; color: #64748b;">Evaluating query clarity and verification criteria...</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        async for chunk in full_agent.astream(
            {"messages": [input_message]},
            config=thread_config,
            stream_mode="updates"
        ):
            for node_name, node_output in chunk.items():
                
                # 1. Scoping / Clarification Check
                if node_name == "clarify_with_user":
                    messages = node_output.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                        
                        if any(k in content.lower() for k in ["clarify", "specify", "could you please", "1.", "2."]) and allow_clarification:
                            step1_box.markdown(
                                f"""
                                <div class="agent-step-card" style="border-left: 4px solid #f59e0b;">
                                    <div class="agent-step-header">
                                        <span>🎯 1. Scoping Agent</span>
                                        <span class="badge badge-warning">Clarification Requested</span>
                                    </div>
                                    <div style="font-size: 13px; color: #1e293b; margin-top: 4px;">{content}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            agent_output["clarification_needed"] = True
                            agent_output["clarification_question"] = content
                            return agent_output
                        else:
                            step1_box.markdown(
                                f"""
                                <div class="agent-step-card" style="border-left: 4px solid #10b981;">
                                    <div class="agent-step-header">
                                        <span>🎯 1. Scoping Agent</span>
                                        <span class="badge badge-success">Scope Verified</span>
                                    </div>
                                    <div style="font-size: 13px; color: #475569;">{content}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # 2. Research Brief Generation
                elif node_name == "write_research_brief":
                    brief = node_output.get("research_brief", "")
                    agent_output["brief"] = brief
                    st.session_state.current_brief = brief
                    
                    step2_box.markdown(
                        f"""
                        <div class="agent-step-card" style="border-left: 4px solid #3b82f6;">
                            <div class="agent-step-header">
                                <span>📋 2. Structured Research Brief</span>
                                <span class="badge badge-success">Generated</span>
                            </div>
                            <div style="font-size: 13px; color: #334155; line-height: 1.5; background: #f8fafc; padding: 10px; border-radius: 8px;">
                                {brief}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    
                    step3_box.markdown(
                        """
                        <div class="agent-step-card">
                            <div class="agent-step-header">
                                <span>⚙️ 3. Supervisor & Parallel Researchers</span>
                                <span class="badge badge-info">Orchestrating search units...</span>
                            </div>
                            <div style="font-size: 13px; color: #64748b;">Decomposing into sub-queries and executing parallel web retrieval via Tavily...</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # 3. Supervisor Subgraph
                elif node_name == "supervisor_subgraph":
                    notes = node_output.get("notes", [])
                    raw_notes = node_output.get("raw_notes", [])
                    agent_output["notes"] = notes
                    st.session_state.current_notes = notes

                    step3_box.markdown(
                        f"""
                        <div class="agent-step-card" style="border-left: 4px solid #8b5cf6;">
                            <div class="agent-step-header">
                                <span>⚙️ 3. Supervisor & Parallel Researchers</span>
                                <span class="badge badge-success">Completed ({len(notes)} evidence units)</span>
                            </div>
                            <div style="font-size: 13px; color: #475569;">
                                Parallel research executed across {len(notes)} key focus areas with web verification.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    step4_box.markdown(
                        """
                        <div class="agent-step-card">
                            <div class="agent-step-header">
                                <span>✍️ 4. Final Report Synthesis</span>
                                <span class="badge badge-info">Generating executive report...</span>
                            </div>
                            <div style="font-size: 13px; color: #64748b;">Structuring comprehensive analysis, comparative metrics, and citations...</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # 4. Final Report Generation
                elif node_name == "final_report_generation":
                    report = node_output.get("final_report", "")
                    agent_output["final_report"] = report
                    agent_output["status"] = "completed"

                    step4_box.markdown(
                        """
                        <div class="agent-step-card" style="border-left: 4px solid #10b981;">
                            <div class="agent-step-header">
                                <span>✍️ 4. Final Report Synthesis</span>
                                <span class="badge badge-success">Finalized</span>
                            </div>
                            <div style="font-size: 13px; color: #166534;">Comprehensive research report successfully compiled.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        return agent_output

    except Exception as e:
        st.error(f"Execution Exception in Multi-Agent Pipeline: {str(e)}")
        agent_output["status"] = "error"
        agent_output["error"] = str(e)
        return agent_output


# ==============================================================================
# TAB 1: AUTONOMOUS RESEARCH STUDIO
# ==============================================================================
with tab_research:
    col_input, col_suggestions = st.columns([3, 1])

    with col_suggestions:
        st.markdown("<p style='font-size: 12px; font-weight: 600; color: #64748b; margin-bottom: 6px;'>SUGGESTED TOPICS</p>", unsafe_allow_html=True)
        suggested = [
            "Best IPL team performance and title analysis",
            "State of Quantum Computing commercialization in 2026",
            "Solid-state battery breakthroughs and commercial hurdles",
        ]
        chosen_prompt = None
        for s in suggested:
            if st.button(f"📌 {s[:35]}...", key=f"sug_{s}", use_container_width=True):
                chosen_prompt = s

    with col_input:
        default_query = chosen_prompt if chosen_prompt else ""
        user_query = st.text_area(
            "Research Query or Topic",
            value=default_query,
            height=100,
            placeholder="Enter the research subject (e.g., 'Perform a comprehensive comparative analysis of the best IPL teams of all time, including win rates, captaincy impact, and squad depth')...",
            help="Be as broad or detailed as you like. If ambiguous and clarification is enabled, the agent will ask clarifying questions.",
        )

    # Start Button & Status
    col_btn, col_stats = st.columns([1, 3])
    with col_btn:
        start_clicked = st.button("🚀 Start Deep Research", type="primary", use_container_width=True)

    with col_stats:
        if st.session_state.thread_id:
            st.markdown(
                f"<div style='font-size: 12px; color: #94a3b8; padding-top: 10px;'>"
                f"Session ID: <code>{st.session_state.thread_id[:8]}...</code> | Active Engine: <code>{supervisor_model}</code>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Handle Clarification Flow if awaiting user input
    if st.session_state.clarification_question:
        st.warning("⚠️ **The Scoping Agent requires additional clarification before initiating research:**")
        st.info(st.session_state.clarification_question)
        
        clarification_response = st.text_input("Your Response / Clarification:", key="clarif_resp_input")
        if st.button("Submit Clarification & Continue Research", type="primary"):
            if clarification_response.strip():
                with st.spinner("Resuming research pipeline with your clarification..."):
                    start_time = time.time()
                    res = asyncio.run(
                        execute_research_pipeline(
                            clarification_response, 
                            st.session_state.thread_id, 
                            is_clarification_response=True
                        )
                    )
                    elapsed = time.time() - start_time
                    if not res.get("clarification_needed"):
                        st.session_state.clarification_question = None
                        st.session_state.research_results = res
                        st.session_state.research_results["elapsed"] = elapsed
                        st.session_state.research_results["query"] = user_query + f" (Clarified: {clarification_response})"
                        # Save to history
                        hist = load_history()
                        hist.insert(0, {
                            "id": str(uuid.uuid4()),
                            "thread_id": st.session_state.thread_id,
                            "query": st.session_state.research_results["query"],
                            "brief": res.get("brief", ""),
                            "final_report": res.get("final_report", ""),
                            "notes": res.get("notes", []),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "elapsed": round(elapsed, 1),
                        })
                        save_history(hist)
                        st.rerun()

    # Trigger New Research
    if start_clicked:
        if not user_query.strip():
            st.warning("Please enter a research topic to begin.")
        elif not gemini_key or not tavily_key:
            st.error("Missing required API keys. Please verify GOOGLE_API_KEY (or GEMINI_API_KEY) and TAVILY_API_KEY in your .env file.")
        else:
            st.session_state.clarification_question = None
            st.session_state.research_results = None
            
            with st.spinner("Orchestrating autonomous multi-agent research pipeline..."):
                start_time = time.time()
                results = asyncio.run(
                    execute_research_pipeline(user_query, st.session_state.thread_id)
                )
                elapsed = time.time() - start_time

                if results.get("clarification_needed"):
                    st.session_state.clarification_question = results.get("clarification_question")
                    st.rerun()
                elif results.get("final_report"):
                    results["elapsed"] = elapsed
                    results["query"] = user_query
                    st.session_state.research_results = results

                    # Save to persistent history
                    hist = load_history()
                    hist.insert(0, {
                        "id": str(uuid.uuid4()),
                        "thread_id": st.session_state.thread_id,
                        "query": user_query,
                        "brief": results.get("brief", ""),
                        "final_report": results.get("final_report", ""),
                        "notes": results.get("notes", []),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "elapsed": round(elapsed, 1),
                    })
                    save_history(hist)
                    st.rerun()

    # Display Completed Research Report
    if st.session_state.research_results and st.session_state.research_results.get("final_report"):
        res = st.session_state.research_results
        
        st.markdown("---")
        
        # Summary KPI Bar
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(
                f"<div class='stat-box'><div class='stat-val'>{res.get('elapsed', 0):.1f}s</div><div class='stat-label'>Execution Time</div></div>",
                unsafe_allow_html=True,
            )
        with col_m2:
            st.markdown(
                f"<div class='stat-box'><div class='stat-val'>{len(res.get('notes', []))}</div><div class='stat-label'>Research Units</div></div>",
                unsafe_allow_html=True,
            )
        with col_m3:
            word_count = len(res.get("final_report", "").split())
            st.markdown(
                f"<div class='stat-box'><div class='stat-val'>{word_count:,}</div><div class='stat-label'>Report Words</div></div>",
                unsafe_allow_html=True,
            )
        with col_m4:
            st.markdown(
                f"<div class='stat-box'><div class='stat-val'>{report_model}</div><div class='stat-label'>Synthesizer</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Download & Action Toolbar
        col_dl, col_blank = st.columns([1, 4])
        with col_dl:
            st.download_button(
                label="📥 Download Report (.md)",
                data=res.get("final_report", ""),
                file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        # Report Content Card
        st.markdown(
            f"""
            <div class="report-container">
                {res.get("final_report", "")}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Collapsible Detailed Findings & Brief
        with st.expander("🔍 View Detailed Extracted Evidence & Research Notes", expanded=False):
            if res.get("brief"):
                st.markdown("#### Structured Research Brief")
                st.info(res.get("brief"))
            if res.get("notes"):
                st.markdown("#### Research Worker Findings")
                for i, note in enumerate(res.get("notes"), start=1):
                    st.markdown(f"**Unit {i} Findings:**")
                    st.write(note)
                    st.markdown("---")


# ==============================================================================
# TAB 2: MULTI-AGENT ARCHITECTURE (GUIDE PRESENTATION)
# ==============================================================================
with tab_architecture:
    st.markdown("### 🏛️ LangGraph Multi-Agent Architecture")
    st.caption("Demonstrating the hierarchical, cyclic multi-agent orchestration to project supervisors")

    st.markdown(
        """
        The **Deep Research Agent** implements a multi-level state machine built on **LangGraph**.
        Unlike simple single-turn prompts or basic RAG, this system divides research into five dedicated specialized agents:
        """
    )

    # Visual Workflow Diagram (Mermaid)
    st.markdown(
        """
        ```mermaid
        graph TD
            START([User Prompt]) --> Scoping[1. Scoping Agent]
            Scoping -->|Ambiguous| Clarify[Clarify with User]
            Clarify -->|User Reply| Scoping
            Scoping -->|Specific & Validated| Brief[2. Write Research Brief]
            Brief --> Supervisor[3. Supervisor Orchestrator]
            
            subgraph Parallel_Research [Parallel Autonomous Exploration]
                Supervisor -->|Topic 1| R1[Researcher Worker 1 + Tavily]
                Supervisor -->|Topic 2| R2[Researcher Worker 2 + Tavily]
                Supervisor -->|Topic N| RN[Researcher Worker N + Tavily]
                R1 -->|Synthesized Notes| Supervisor
                R2 -->|Synthesized Notes| Supervisor
                RN -->|Synthesized Notes| Supervisor
            end
            
            Supervisor -->|Research Complete| Synthesis[4. Final Report Generation]
            Synthesis --> END([Deliver Executive Report])
            
            style START fill:#0f172a,stroke:#334155,stroke-width:2px,color:#fff
            style Scoping fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0369a1
            style Brief fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#15803d
            style Supervisor fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#92400e
            style Parallel_Research fill:#f8fafc,stroke:#cbd5e1,stroke-width:2px
            style Synthesis fill:#f3e8ff,stroke:#9333ea,stroke-width:2px,color:#6b21a8
            style END fill:#0f172a,stroke:#334155,stroke-width:2px,color:#fff
        ```
        """
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Detailed agent specifications for academic presentation
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown(
            """
            #### 1. 🎯 Scoping Agent
            - **Role**: Ensures deterministic requirement validation using Pydantic structured output (`ClarifyWithUser`).
            - **Decision Loop**: If requirements are vague (e.g. *"tell me about AI"*), halts and prompts the user for specific parameters rather than hallucinating.
            - **Guarantees**: Prevents uncontrolled API token waste on ambiguous topics.

            #### 2. 📋 Research Brief Formulator
            - **Role**: Converts unstructured conversational exchanges into a formalized research document with explicit hypotheses, target metrics, and temporal boundaries (`get_today_str()`).
            - **Output**: Transferred directly to the Supervisor.
            """
        )

    with col_a2:
        st.markdown(
            """
            #### 3. ⚙️ Supervisor Orchestrator & Parallel Subgraph
            - **Role**: Lead researcher that dynamically decomposes the brief into parallel research tracks (`ConductResearch`).
            - **Concurrency**: Asynchronously launches multiple `researcher_agent` instances concurrently (`asyncio.gather`), each empowered with Tavily search and ReAct reasoning tools.
            - **ReAct Control**: Governed by iteration caps and `ResearchComplete` stopping criteria.

            #### 4. ✍️ Final Report Synthesizer
            - **Role**: Ingests all deduplicated notes and compiles an academic-grade report complete with executive summary, comparative tables, and citations.
            """
        )


# ==============================================================================
# TAB 3: SESSION HISTORY
# ==============================================================================
with tab_history:
    st.markdown("### 📜 Past Research Reports")
    history_data = load_history()

    if not history_data:
        st.info("No research sessions recorded yet. Run your first research topic in the studio above!")
    else:
        for idx, item in enumerate(history_data):
            with st.expander(f"📄 {item.get('query', 'Untitled')[:70]} ({item.get('timestamp', 'Recent')})", expanded=(idx == 0)):
                col_h1, col_h2 = st.columns([4, 1])
                with col_h1:
                    st.caption(f"Execution Time: {item.get('elapsed', 0)}s | Findings: {len(item.get('notes', []))} units")
                with col_h2:
                    st.download_button(
                        "📥 Download .md",
                        data=item.get("final_report", ""),
                        file_name=f"report_{item.get('id', 'session')[:6]}.md",
                        mime="text/markdown",
                        key=f"dl_hist_{item.get('id', idx)}",
                        use_container_width=True,
                    )

                st.markdown(
                    f"""
                    <div class="report-container" style="padding: 20px 24px; margin-top: 10px;">
                        {item.get("final_report", "")}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
