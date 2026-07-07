# LangGraph Playground

A hands-on collection of **LangGraph** projects and notebooks exploring agentic workflows, retrieval-augmented generation (RAG), memory, human-in-the-loop (HITL) control, streaming, subgraphs, and the Model Context Protocol (MCP) — built primarily with **LangChain**, **LangGraph**, and **Groq**-hosted LLMs.

![Jupyter Notebook](https://img.shields.io/badge/Jupyter%20Notebook-94.6%25-orange)
![Python](https://img.shields.io/badge/Python-3.7%25-blue)
![HTML](https://img.shields.io/badge/HTML-1.7%25-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

This repository is a personal learning lab for building stateful, graph-based LLM applications with [LangGraph](https://github.com/langchain-ai/langgraph). It walks through core LangGraph concepts step by step — from basic graphs and tool calling to more advanced patterns like corrective/self-reflective RAG, subgraphs, checkpointed memory, and human-in-the-loop workflows — using a mix of Jupyter notebooks and standalone Python scripts.

## Features

- **Corrective RAG** — a retrieval pipeline that grades and corrects retrieved documents before generation (`Corrective-RAG/`)
- **Self-RAG** — a self-reflective retrieval-augmented generation pattern where the model critiques its own retrieval and output (`Self-RAG/`)
- **Human-in-the-Loop (HITL)** — interrupting and resuming graph execution for human review/approval (`HITL.py`, `hitl_simple.py`)
- **Short-term & long-term memory** — checkpointed conversational state and persistent memory across sessions (`shorttermmemory/`, `longtermmemory/`)
- **Chatbots** — a chatbot with a web UI and a chatbot backed by a database (`chatbotwithui/`, `chatbotwithdatabase.py`)
- **Streaming** — streaming LLM/graph outputs token-by-token or event-by-event (`streaming.py`)
- **Subgraphs** — composing graphs from smaller, reusable subgraphs, including shared-state patterns (`subgraph.py`, `subgraphSamestate.py`)
- **Tool calling** — custom tool definitions for agents (`Tools.py`)
- **Model Context Protocol (MCP)** — integrating MCP servers/clients into a LangGraph agent (`mcp_folder/`)
- **LangSmith tracing** — observability and debugging of graph runs (`Langsmith/`)
- **Reference notebooks & materials** — exploratory notebooks and supporting reading material (`Notebooks/`, `books/`)

## Tech Stack

| Category | Tools / Libraries |
|---|---|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain) (`langchain-core`, `langchain-community`, `langchain-text-splitters`) |
| LLM Provider | [Groq](https://groq.com/) via `langchain-groq` |
| Persistence | `langgraph-checkpoint-sqlite` (SQLite-backed checkpointing) |
| Retrieval / RAG | `faiss-cpu`, `sentence-transformers`, `pypdf` |
| Web search tool | `ddgs` (DuckDuckGo search) |
| Agent tooling | `mcp` (Model Context Protocol) |
| API / serving | `fastapi`, `uvicorn` |
| Config | `python-dotenv` |
| Observability | LangSmith |

## Repository Structure

```
LangGraph/
├── Corrective-RAG/       # Corrective RAG pipeline (grades & corrects retrieved docs)
├── Self-RAG/             # Self-reflective RAG pattern
├── Langsmith/            # LangSmith tracing / observability examples
├── Notebooks/            # Exploratory Jupyter notebooks on LangGraph concepts
├── books/                # Reference/supporting reading material
├── chatbotwithui/        # Chatbot with a web-based UI
├── longtermmemory/       # Persistent, cross-session memory examples
├── shorttermmemory/      # Checkpointed, in-session memory examples
├── mcp_folder/           # Model Context Protocol integration
├── HITL.py               # Human-in-the-loop graph interrupt/resume
├── hitl_simple.py         # Minimal HITL example
├── Tools.py               # Custom tool definitions for agents
├── chatbotwithdatabase.py # Chatbot backed by a database
├── streaming.py           # Streaming graph/LLM outputs
├── subgraph.py             # Composing graphs from subgraphs
├── subgraphSamestate.py    # Subgraphs sharing state with the parent graph
├── requirement.txt        # Python dependencies
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+ (recommended)
- A [Groq API key](https://console.groq.com/) (used by `langchain-groq`)
- Jupyter Notebook or JupyterLab (for the notebook-based examples)

### Installation

Clone the repository:

```bash
git clone https://github.com/AmirBR996/LangGraph.git
cd LangGraph
```

Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirement.txt
```

### Configuration

Create a `.env` file in the project root with the API keys required by the examples you want to run, for example:

```env
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key      # optional, for LangSmith tracing
LANGCHAIN_TRACING_V2=true                      # optional, for LangSmith tracing
```

### Usage

**Run a standalone script**, e.g. the streaming or HITL examples:

```bash
python streaming.py
python HITL.py
```

**Run the FastAPI-backed chatbot:**

```bash
uvicorn chatbotwithdatabase:app --reload
```

**Explore the notebooks:**

```bash
jupyter notebook
```

Then open any notebook under `Notebooks/`, `Corrective-RAG/`, or `Self-RAG/` to follow along interactively.

## Contributing

This is primarily a personal learning repository, but suggestions, issues, and pull requests are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m "Add my feature"`)
4. Push the branch (`git push origin feature/my-feature`)
5. Open a pull request

## License

Distributed under the MIT License. See `LICENSE` for details.

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph) for the core orchestration framework
- [Groq](https://groq.com/) for fast LLM inference
- The broader open-source RAG/agent community whose patterns (Corrective RAG, Self-RAG) inspired parts of this repo
