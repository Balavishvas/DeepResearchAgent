from datetime import datetime
from typing import List, Literal, Annotated
from tavily import TavilyClient
from langchain_google_genai import ChatGoogleGenerativeAI
from deep_research.state import Summary
from deep_research.prompt import summarize_webpage_prompt
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import os
import time
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_SUMMARIZATION_MODEL = "gemini-3.5-flash-lite"

def get_today_str() -> str:
    """Get current date formatted for display in prompts and outputs.

    Returns:
        Human-readable date string in format like 'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"

def get_summarization_model(model_name: str = _DEFAULT_SUMMARIZATION_MODEL) -> ChatGoogleGenerativeAI:
    """Get the summarization model instance."""
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=1.0,
        max_tokens=None,
        timeout=None
    )

# Lazy singletons — created on first use so importing utils never fails
_summarization_model = None
_tavily_client = None


def _get_or_create_summarization_model():
    """Return the module-level summarization model, creating it lazily."""
    global _summarization_model
    if _summarization_model is None:
        _summarization_model = get_summarization_model()
    return _summarization_model


def _get_or_create_tavily_client():
    """Return the module-level Tavily client, creating it lazily."""
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient()
    return _tavily_client


# Keep backward-compatible module-level names as properties via a helper
def __getattr__(name):
    if name == "summarization_model":
        return _summarization_model
    if name == "tavily_client":
        return _tavily_client
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
# ===== SEARCH FUNCTIONS =====

def tavily_search_multiple(
    search_queries: List[str], 
    max_results: int = 5, 
    topic: Literal["general", "news", "finance"] = "general", 
    include_raw_content: bool = True, 
) -> List[dict]:
    """Perform search using Tavily API for multiple queries.

    Args:
        search_queries: List of search queries to execute
        max_results: Maximum number of results per query
        topic: Topic filter for search results
        include_raw_content: Whether to include raw webpage content

    Returns:
        List of search result dictionaries
    """
    
    # Execute searches sequentially. Note: you can use AsyncTavilyClient to parallelize this step.
    client = _get_or_create_tavily_client()
    search_docs = []
    for query in search_queries:
        result = client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )
        search_docs.append(result)

    return search_docs

def summarize_webpage_content(webpage_content: str, model_name: str = _DEFAULT_SUMMARIZATION_MODEL) -> str:
    """Summarize webpage content using the configured summarization model.

    Args:
        webpage_content: Raw webpage content to summarize
        model_name: LangChain model string (provider:model-id)

    Returns:
        Formatted summary with key excerpts
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            model = _get_or_create_summarization_model()
            structured_model = model.with_structured_output(Summary)

            # Generate summary
            summary = structured_model.invoke([
                HumanMessage(content=summarize_webpage_prompt.format(
                    webpage_content=webpage_content,
                    date=get_today_str()
                ))
            ])

            # Format summary with clear structure
            formatted_summary = (
                f"<summary>\n{summary.summary}\n</summary>\n\n"
                f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
            )
            return formatted_summary

        except Exception as e:
            err_str = str(e)
            # Handle rate limit (429) with exponential backoff
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait_time = 30 * (attempt + 1)  # 30s, 60s, 90s
                print(f"Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    print("Max retries reached. Falling back to raw content.")
                    return webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content
            else:
                print(f"Failed to summarize webpage: {err_str}")
                return webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content

      
def deduplicate_search_results(search_results: List[dict]) -> dict:
    """Deduplicate search results by URL to avoid processing duplicate content.
    
    Args:
        search_results: List of search result dictionaries
        
    Returns:
        Dictionary mapping URLs to unique results
    """
    unique_results = {}
    
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = result
    
    return unique_results

def process_search_results(unique_results: dict) -> dict:
    """Process search results by summarizing content where available.
    
    Args:
        unique_results: Dictionary of unique search results
        
    Returns:
        Dictionary of processed results with summaries
    """
    summarized_results = {}
    
    for url, result in unique_results.items():
        # Use existing content if no raw content for summarization
        if not result.get("raw_content"):
            content = result['content']
        else:
            # Summarize raw content for better processing
            content = summarize_webpage_content(result['raw_content'])
        
        summarized_results[url] = {
            'title': result['title'],
            'content': content
        }
    
    return summarized_results

def format_search_output(summarized_results: dict) -> str:
    """Format search results into a well-structured string output.
    
    Args:
        summarized_results: Dictionary of processed search results
        
    Returns:
        Formatted string of search results with clear source separation
    """
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."
    
    formatted_output = "Search results: \n\n"
    
    for i, (url, result) in enumerate(summarized_results.items(), 1):
        formatted_output += f"\n\n--- SOURCE {i}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "-" * 80 + "\n"
    
    return formatted_output

from langchain_core.tools import tool, InjectedToolArg

@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """Fetch results from Tavily search API with content summarization.

    Args:
        query: A single search query to execute
        max_results: Maximum number of results to return
        topic: Topic to filter results by ('general', 'news', 'finance')

    Returns:
        Formatted string of search results with summaries
    """

    search_results = tavily_search_multiple(
        [query],
        max_results=max_results,
        topic=topic,
        include_raw_content=False,
    )

    unique_results = deduplicate_search_results(search_results)

    summarized_results = process_search_results(unique_results)
    return format_search_output(summarized_results)

@tool(parse_docstring=False)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.
    
    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.
    
    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?
    
    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?
    
    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps
        
    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"


def format_messages(messages: list) -> None:
    """Format and display a list of LangChain messages nicely in Jupyter notebooks.

    Args:
        messages: List of LangChain message objects to display
    """
    import json

    def _extract_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for p in content:
                if isinstance(p, dict) and "text" in p:
                    texts.append(p["text"])
                elif hasattr(p, "text"):
                    texts.append(p.text)
                else:
                    texts.append(str(p))
            return "\n".join(texts)
        return str(content)

    for msg in messages:
        msg_type = getattr(msg, "type", msg.__class__.__name__)
        header = msg_type.upper()

        print(f"\n{'=' * 30} {header} {'=' * 30}\n")

        # Display tool calls if present on AI messages
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                name = tc.get("name", "Tool")
                args = tc.get("args", {})
                call_id = tc.get("id", "")
                print(f"Tool Call: {name}")
                if call_id:
                    print(f"  ID: {call_id}")
                print(f"  Args: {json.dumps(args, indent=2)}")

        # Display content
        content = getattr(msg, "content", "")
        text = _extract_text(content).strip()
        if text:
            name = getattr(msg, "name", "")
            if name:
                print(f"[{name}]")
            print(text)
