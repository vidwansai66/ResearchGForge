# ResearchForge AI 🔬

**AI Research & Evidence Synthesis Agent**

ResearchForge AI is a specialized AI agent designed for college evaluations, built using LangGraph, LangServe, and Google Gemini. It autonomously synthesizes research reports based on a user's query by identifying the research type, drafting a plan, extracting offline knowledge through RAG (Retrieval-Augmented Generation), analyzing the evidence, and formatting a final structured report.

## Features

- **Agentic Workflow**: Uses LangGraph's `StateGraph` to construct an end-to-end multi-step AI workflow.
- **RAG Architecture**: Leverages Gemini Embeddings and an `InMemoryVectorStore` to retrieve relevant chunks of a locally curated enterprise AI knowledge base.
- **Deterministic Python Nodes**: Optimizes Gemini text-generation API quotas by executing deterministic offline analysis for query classification, plan generation, and evidence extraction.
- **Quota & Error Handling**: Gracefully intercepts `429 RESOURCE_EXHAUSTED` responses and renders offline extraction results directly in the UI.
- **FastAPI / LangServe Backend**: Exposes clean endpoints for the agent and LangServe playground.

## Architecture

1. **Analyze Question**: Evaluates the research topic and categorizes the research type (e.g. Comparison, Advantages & Limitations, Explanations) without using LLM quota.
2. **Create Research Plan**: Generates a dynamic 4-5 step checklist of the analysis pipeline.
3. **Retrieve Evidence**: Vectorizes the query using Google Gemini embeddings to surface similar concepts.
4. **Analyze Evidence**: Offline regex extraction processes the retrieved documents to map out key findings, applications, advantages, and limitations.
5. **Generate Report**: The only node designed to invoke the Gemini API, formulating a highly structured markdown synthesis of the provided state.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/vidwansai66/ResearchGForge.git
cd ResearchGForge
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your Gemini API key in your environment variables:
```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your-api-key"

# Linux / Mac
export GEMINI_API_KEY="your-api-key"
```

## Running the Application

Start the local server using Uvicorn:

```bash
uvicorn app:app --reload
```

Then visit:
- **Web UI**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **LangServe Playground**: [http://127.0.0.1:8000/researchforge/playground/](http://127.0.0.1:8000/researchforge/playground/)
- **Architecture Spec**: [http://127.0.0.1:8000/architecture](http://127.0.0.1:8000/architecture)
- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

## Project Constraints
Designed for a college viva presentation. The entire backend routing, RAG configuration, AI agent tools, prompt synthesis, and HTML/CSS/JS frontend are unified elegantly into a single `app.py` file to maintain maximal simplicity.
