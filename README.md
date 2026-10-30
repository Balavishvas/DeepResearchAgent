# 🔬 Deep Research Agent

An autonomous multi-agent research system built with **LangGraph**, **Tavily Search**, and selectable **Ollama (local)** or **Google Gemini (cloud)** models.

## Overview

The application coordinates specialized agents to clarify a research question, create a research brief, conduct parallel web research, summarize source pages, compress findings, and generate a citation-backed report.

## Agent workflow

1. **Scoping agent** — identifies ambiguity and asks clarification questions.
2. **Research brief agent** — turns the user request into a focused research plan.
3. **Supervisor agent** — decomposes the plan and coordinates researcher workers.
4. **Parallel researcher agents** — use Tavily search and reasoning tools to gather evidence.
5. **Webpage summarization and compression** — condense source material and research notes.
6. **Final report agent** — synthesizes the evidence into a structured Markdown report.

All LLM calls use the centralized `deep_research/llm.py` model factory. The same provider and model selection is propagated through the graph, including parallel researcher calls and webpage summarization.

## Supported model providers

| Provider | Runs where | Example model |
|---|---|---|
| Ollama | Your local machine | `qwen2.5:7b` |
| Google Gemini | Google cloud API | `gemini-2.5-flash` |

Choose the provider and model in the Streamlit sidebar. Ollama is the default so the project can run locally without a paid LLM API. Gemini requires a Google AI Studio API key.

> Tavily is the web-search service and remains an external API dependency in both modes. Ollama makes model inference local; it does not make web search offline.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) or Python `venv`
- [Tavily API key](https://app.tavily.com) for web research
- For local inference: [Ollama](https://ollama.com/download)
- For cloud inference: [Google AI Studio API key](https://aistudio.google.com/app/apikey)

## Installation

```bash
git clone https://github.com/Balavishvas/DeepResearchAgent.git
cd DeepResearchAgent
uv sync
```

Create your environment file:

**Windows PowerShell**
```powershell
Copy-Item .env.example .env
```

**macOS / Linux**
```bash
cp .env.example .env
```

Add your Tavily key to `.env`:

```env
TAVILY_API_KEY=your_tavily_api_key
```

## Run with Ollama (local)

1. Install and start Ollama.
2. Pull a supported model:

```bash
ollama pull qwen2.5:7b
```

3. Configure `.env`:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
```

4. Start the application:

```bash
uv run streamlit run app.py
```

Ollama must be running while the application is using local inference. Select **Ollama (Local)** in the sidebar and choose a model that has been pulled.

## Run with Google Gemini (cloud)

1. Add your Google AI Studio key to `.env`:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
```

2. Run the application:

```bash
uv run streamlit run app.py
```

Select **Google Gemini (Cloud)** in the sidebar. The API key is read from the environment and is never entered into the UI.

## Switch providers

You can switch between Ollama and Gemini from the Streamlit sidebar without changing agent code. Each agent receives the selected provider and model through LangGraph runtime configuration.

## Configuration reference

| Variable | Purpose | Default |
|---|---|---|
| `LLM_PROVIDER` | `ollama` or `gemini` | `ollama` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Default local model | `qwen2.5:7b` |
| `GEMINI_API_KEY` | Google Gemini API key | — |
| `GEMINI_MODEL` | Default Gemini model | `gemini-2.5-flash` |
| `TAVILY_API_KEY` | Tavily web search key | — |

## Project structure

```
DeepResearchAgent/
├── app.py
├── deep_research/
│   ├── agents/
│   │   ├── scoping_agent.py
│   │   ├── supervisor_agent.py
│   │   ├── research_agent.py
│   │   └── full_agent.py
│   ├── llm.py                 # Centralized Ollama/Gemini model factory
│   ├── prompt.py
│   ├── state.py
│   └── utils.py               # Tavily search and webpage summarization
├── notebooks/
├── .env.example
├── pyproject.toml
└── README.md
```

## Notes

- Never commit your real `.env` file or API keys.
- Tool calling and structured output support can vary by Ollama model. Use a capable instruct model for the supervisor and researcher agents.
- The selected Ollama model must be available locally before starting a research run.
- Gemini usage is subject to Google AI Studio model availability, quotas, and billing policies.

## License

MIT
