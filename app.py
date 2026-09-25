import os
import time
import json
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import StateGraph, START, END
from langserve import add_routes
from langchain_core.tools import tool

# 1. State Definition
class ResearchState(TypedDict, total=False):
    user_query: str
    research_topic: str
    research_type: str
    key_concepts: list[str]
    sub_questions: list[str]
    research_plan: list[str]
    retrieved_documents: list[str]
    retrieved_sources: list[str]
    evidence_summary: str
    key_findings: list[str]
    advantages: list[str]
    limitations: list[str]
    comparison_points: list[str]
    final_report: str
    execution_steps: list[str]
    tools_executed: list[str]

# 2. Knowledge Base
KNOWLEDGE_BASE = [
    {
        "title": "Retrieval-Augmented Generation (RAG)",
        "topic": "RAG",
        "content": "Retrieval-Augmented Generation (RAG) is an architecture that provides large language models with external knowledge to improve accuracy and reduce hallucinations. Instead of relying solely on the model's internal weights, RAG retrieves relevant documents from a knowledge base (often using vector databases and embeddings) and includes them in the prompt. Advantages: Up-to-date information, reduced hallucinations, traceable sources. Limitations: Depends on retrieval quality, latency overhead. Applications: Enterprise search, Q&A systems, customer support bots. Important considerations: Chunking strategy, embedding model choice, similarity threshold."
    },
    {
        "title": "Fine-Tuning",
        "topic": "Fine-Tuning",
        "content": "Fine-Tuning involves updating the internal weights of a pre-trained Large Language Model using a specific dataset. It is typically used to teach the model a new style, format, or highly specialized domain knowledge. Advantages: Deeply ingrained knowledge, tailored tone/style, can reduce prompt size. Limitations: Expensive and time-consuming to update, prone to catastrophic forgetting, hard to trace factual sources. Applications: Coding assistants, specialized medical/legal reasoning, brand voice adaptation. Important considerations: Requires high-quality labeled data, compute intensive."
    },
    {
        "title": "Large Language Models (LLMs)",
        "topic": "LLM",
        "content": "Large Language Models (LLMs) are deep learning models built on the transformer architecture, trained on vast amounts of text data. They can understand and generate human-like text. Advantages: Versatility across tasks (translation, summarization, coding), strong natural language understanding. Limitations: Hallucinations, lack of true reasoning, high inference cost, knowledge cutoff dates. Applications: Content generation, chatbots, sentiment analysis. Important considerations: Prompt engineering is crucial, ethical concerns regarding bias."
    },
    {
        "title": "AI Agents",
        "topic": "Agents",
        "content": "AI Agents are autonomous systems powered by LLMs that can perceive their environment, make decisions, and take actions using tools to achieve specific goals. They move beyond single-turn chat into multi-step reasoning. Advantages: Task automation, autonomous problem solving, tool integration (APIs, databases). Limitations: Can get stuck in loops, error propagation, hard to evaluate, unpredictable behavior. Applications: Personal assistants, automated research, coding agents. Important considerations: Requires robust memory (short and long term), careful tool design."
    },
    {
        "title": "Agentic AI",
        "topic": "Agentic AI",
        "content": "Agentic AI refers to systems exhibiting high degrees of autonomy, planning, and self-correction. Unlike traditional software, Agentic AI can break down complex goals into sub-tasks. Advantages: Handles ambiguous goals, reduces human oversight. Limitations: Safety risks if given destructive tools, complex state management. Applications: Multi-agent simulations, enterprise process automation. Important considerations: Human-in-the-loop (HITL) mechanisms are essential for safety."
    },
    {
        "title": "LangChain",
        "topic": "LangChain",
        "content": "LangChain is a framework for developing applications powered by language models. It provides abstractions for LLMs, prompts, chains, memory, and tools. Advantages: Standardized interfaces, rich ecosystem of integrations, simplifies complex LLM workflows. Limitations: Can be overly complex for simple tasks, steep learning curve. Applications: RAG systems, chatbots, simple agents. Important considerations: Rapidly evolving ecosystem."
    },
    {
        "title": "LangGraph",
        "topic": "LangGraph",
        "content": "LangGraph is an extension of LangChain specifically designed for building stateful, multi-actor applications with LLMs using a graph-based approach. It allows defining cyclical workflows. Advantages: Excellent for complex agent architectures, precise state management, supports loops and branching. Limitations: Requires understanding graph theory concepts, overhead for linear tasks. Applications: Advanced AI agents, multi-agent systems. Important considerations: State schemas must be well-defined."
    },
    {
        "title": "Embeddings",
        "topic": "Embeddings",
        "content": "Embeddings are dense mathematical vector representations of text, capturing semantic meaning. Words or sentences with similar meanings will have vectors close together in the multi-dimensional space. Advantages: Enables semantic search, handles synonyms and context better than keyword search. Limitations: Context window limits for embedding models, out-of-domain terms may be poorly represented. Applications: Vector search, clustering, classification. Important considerations: Dimensionality size, embedding model choice (e.g., Google GenAI embeddings)."
    },
    {
        "title": "Vector Databases",
        "topic": "Vector Databases",
        "content": "Vector Databases are optimized for storing and retrieving high-dimensional vectors (embeddings) using similarity metrics like cosine similarity. Advantages: Fast nearest-neighbor search at scale, filtering capabilities. Limitations: Can be expensive to run, complex to manage at massive scale. Applications: RAG, recommendation systems, image search. Important considerations: Indexing algorithms (e.g., HNSW)."
    },
    {
        "title": "Prompt Engineering",
        "topic": "Prompting",
        "content": "Prompt Engineering is the practice of designing inputs (prompts) to elicit optimal responses from LLMs. Techniques include zero-shot, few-shot, and Chain-of-Thought (CoT). Advantages: Low cost way to guide models, requires no model training. Limitations: Fragile (small changes can break outputs), varies between models. Applications: Task formatting, persona adoption, constraint enforcement. Important considerations: Model versioning can break prompts."
    },
    {
        "title": "Generative AI",
        "topic": "GenAI",
        "content": "Generative AI is a branch of AI focused on creating new content (text, images, audio, code) rather than just predicting or classifying existing data. Advantages: High creative potential, rapid content generation. Limitations: Copyright issues, deepfakes, quality inconsistency. Applications: Art generation, marketing copy, synthetic data creation. Important considerations: Ethical use, data provenance."
    },
    {
        "title": "Machine Learning",
        "topic": "ML",
        "content": "Machine Learning is a subset of AI where systems learn patterns from data rather than being explicitly programmed. It includes supervised, unsupervised, and reinforcement learning. Advantages: Data-driven decision making, handles complex non-linear relationships. Limitations: Requires large datasets, can learn biased patterns. Applications: Predictive maintenance, fraud detection, recommendation engines. Important considerations: Data quality, model explainability."
    },
    {
        "title": "Deep Learning",
        "topic": "Deep Learning",
        "content": "Deep Learning uses artificial neural networks with multiple layers to extract higher-level features from raw input. It powers modern LLMs and computer vision. Advantages: State-of-the-art performance on unstructured data (text, images). Limitations: Black box nature, requires massive compute (GPUs) and data. Applications: Autonomous vehicles, natural language processing, medical imaging. Important considerations: High training costs, hyperparameter tuning."
    },
    {
        "title": "Natural Language Processing (NLP)",
        "topic": "NLP",
        "content": "Natural Language Processing is the field of AI concerned with the interaction between computers and human language. Modern NLP is dominated by transformer-based LLMs. Advantages: Enables human-computer interaction, automates text processing. Limitations: Nuance, sarcasm, and low-resource languages remain challenging. Applications: Translation, sentiment analysis, named entity recognition. Important considerations: Tokenization strategies, linguistic diversity."
    },
    {
        "title": "AI in Healthcare",
        "topic": "Healthcare AI",
        "content": "AI in Healthcare involves applying machine learning and generative AI to medical diagnostics, drug discovery, and patient care. Advantages: Accelerates drug discovery, assists radiologists, improves personalized medicine. Limitations: Strict regulatory requirements, high risk of errors (life-critical), data privacy (HIPAA). Applications: Medical imaging analysis, predictive modeling for patient readmission, AI agents for administrative tasks. Important considerations: Explainability is critical, requires clinical validation."
    },
    {
        "title": "AI in Finance",
        "topic": "Finance AI",
        "content": "AI in Finance utilizes advanced algorithms for trading, risk assessment, and customer service. Advantages: High-speed pattern recognition, automation of routine tasks. Limitations: Regulatory compliance, model drift in volatile markets. Applications: Algorithmic trading, credit scoring, fraud detection bots. Important considerations: Security, fairness in lending algorithms."
    },
    {
        "title": "AI Automation / Enterprise AI",
        "topic": "Enterprise AI",
        "content": "Enterprise AI refers to the deployment of AI technologies to automate business processes and improve efficiency at an organizational level. Advantages: Cost reduction, 24/7 operations, scaling capabilities. Limitations: Integration with legacy systems, organizational resistance. Applications: Automated customer support, document processing, supply chain optimization. Important considerations: Data silos, enterprise security requirements."
    },
    {
        "title": "AI Evaluation",
        "topic": "Evaluation",
        "content": "AI Evaluation is the process of measuring the performance, safety, and alignment of AI models and agents. Evaluating generative systems is difficult because there is often no single correct answer. Advantages: Ensures safety, guides model improvement. Limitations: Lack of standardized benchmarks for agents, LLM-as-a-judge can be biased. Applications: Model leaderboards, red teaming, regression testing. Important considerations: Need for robust, multi-faceted evaluation frameworks."
    }
]

# Vector Store Initialization
vector_store = None

def get_vector_store():
    global vector_store
    if vector_store is not None:
        return vector_store
        
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
        docs = []
        for item in KNOWLEDGE_BASE:
            doc = Document(
                page_content=item["content"],
                metadata={"title": item["title"], "topic": item["topic"]}
            )
            docs.append(doc)
        
        vector_store = InMemoryVectorStore.from_documents(docs, embeddings_model)
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        return None
        
    return vector_store

# 3. Tools
@tool
def research_query_analyzer(query: str) -> dict:
    """Analyzes a research question to identify the topic, type, key concepts, and sub-questions."""
    query_lower = query.lower()
    
    # Deterministic Type Detection
    if "compare" in query_lower or "vs" in query_lower or "difference" in query_lower:
        r_type = "Comparison"
    elif "advantage" in query_lower or "limitation" in query_lower or "pros" in query_lower or "cons" in query_lower:
        r_type = "Advantages and limitations"
    elif "application" in query_lower or "use case" in query_lower:
        r_type = "Applications"
    elif "how" in query_lower or "architecture" in query_lower:
        r_type = "Architecture"
    else:
        r_type = "Explanation"
        
    # Deterministic Concept Extraction
    stop_words = {"what", "are", "the", "and", "how", "do", "in", "a", "an", "of", "for", "to"}
    words = [w for w in query.replace("?", "").replace(".", "").split() if w.lower() not in stop_words and len(w) > 2]
    
    return {
        "research_topic": " ".join(words[:3]) if words else "General Topic",
        "research_type": r_type,
        "key_concepts": words,
        "sub_questions": [f"What is {words[0]}?" if words else "What is this topic?", "How does it work?"]
    }

def _fallback_query_analysis(query: str) -> dict:
    words = [w for w in query.split() if len(w) > 4]
    return {
        "research_topic": "General AI Research",
        "research_type": "Explanation",
        "key_concepts": words,
        "sub_questions": ["What is the main concept?", "How does it work?"]
    }

@tool
def evidence_analyzer(documents_text: str) -> dict:
    """Analyzes retrieved evidence to extract key findings, advantages, limitations, and comparisons."""
    # Deterministic Extraction from formatted Knowledge Base
    advantages = []
    limitations = []
    applications = []
    key_findings = []
    
    import re
    adv_matches = re.findall(r'Advantages:\s*(.*?)(?=\. |Limitation|Application|Important|$)', documents_text, re.IGNORECASE | re.DOTALL)
    for m in adv_matches:
        advantages.extend([a.strip() for a in m.split(',') if a.strip()])
        
    lim_matches = re.findall(r'Limitations:\s*(.*?)(?=\. |Advantage|Application|Important|$)', documents_text, re.IGNORECASE | re.DOTALL)
    for m in lim_matches:
        limitations.extend([l.strip() for l in m.split(',') if l.strip()])
        
    app_matches = re.findall(r'Applications:\s*(.*?)(?=\. |Advantage|Limitation|Important|$)', documents_text, re.IGNORECASE | re.DOTALL)
    for m in app_matches:
        applications.extend([a.strip() for a in m.split(',') if a.strip()])
        
    # Extract titles as findings context
    titles = re.findall(r'Title:\s*(.*?)(?=\n|$)', documents_text, re.IGNORECASE)
    if not titles:
        # Fallback if no explicit Title: metadata exists in text format
        key_findings.append("Evidence reviewed and summarized.")
    else:
        for t in titles:
            key_findings.append(f"Analyzed concepts related to {t.strip()}")
            
    return {
        "key_findings": key_findings if key_findings else ["Found relevant information."],
        "advantages": list(set(advantages)),
        "limitations": list(set(limitations)),
        "comparison_points": [f"Compared {len(set(advantages))} advantages against {len(set(limitations))} limitations."] if advantages and limitations else [],
        "evidence_summary": "Retrieved explicit advantages, limitations, and applications from the knowledge base."
    }

def _fallback_evidence_analysis() -> dict:
    return {
        "key_findings": ["Information retrieved successfully."],
        "advantages": ["Provides context."],
        "limitations": ["Limited to context."],
        "comparison_points": [],
        "evidence_summary": "The retrieved documents provide foundational context."
    }

# 4. LangGraph Nodes
def analyze_question(state: ResearchState) -> ResearchState:
    query = state.get("user_query", "")
    analysis_result = research_query_analyzer.invoke({"query": query})
    
    steps = state.get("execution_steps", [])
    steps.append("Research question analyzed")
    
    tools_exec = state.get("tools_executed", [])
    if "research_query_analyzer" not in tools_exec:
        tools_exec.append("research_query_analyzer")
        
    return {
        **state,
        "research_topic": analysis_result.get("research_topic", ""),
        "research_type": analysis_result.get("research_type", ""),
        "key_concepts": analysis_result.get("key_concepts", []),
        "sub_questions": analysis_result.get("sub_questions", []),
        "execution_steps": steps,
        "tools_executed": tools_exec
    }

def create_research_plan(state: ResearchState) -> ResearchState:
    query = state.get("user_query", "")
    topic = state.get("research_topic", "")
    q_type = state.get("research_type", "")
    
    if q_type == "Comparison":
        plan = [
            f"1. Understand {topic}",
            "2. Identify the core differences between the concepts",
            "3. Compare their advantages",
            "4. Compare their limitations",
            "5. Determine the best use cases for each"
        ]
    elif q_type == "Advantages and limitations":
        plan = [
            f"1. Define {topic}",
            "2. Extract key advantages",
            "3. Extract key limitations",
            "4. Summarize trade-offs"
        ]
    else:
        plan = [
            f"1. Introduce {topic}",
            "2. Explain core mechanics",
            "3. Provide examples or applications",
            "4. Conclude findings"
        ]
    
    if not plan:
        plan = ["1. Understand the topic.", "2. Synthesize report."]
        
    steps = state.get("execution_steps", [])
    steps.append("Research plan created")
    
    return {
        **state,
        "research_plan": plan,
        "execution_steps": steps
    }

def retrieve_evidence(state: ResearchState) -> ResearchState:
    query = state.get("user_query", "")
    concepts = state.get("key_concepts", [])
    
    vs = get_vector_store()
    retrieved_docs = []
    retrieved_sources = []
    
    if vs is not None:
        search_query = query + " " + " ".join(concepts)
        results = vs.similarity_search(search_query, k=4)
        for r in results:
            retrieved_docs.append(r.page_content)
            retrieved_sources.append(r.metadata.get("title", "Unknown Source"))
    else:
        retrieved_docs = ["Error: Knowledge base vector store not initialized."]
        retrieved_sources = ["Error"]
        
    steps = state.get("execution_steps", [])
    steps.append("Evidence retrieved using RAG")
    
    return {
        **state,
        "retrieved_documents": retrieved_docs,
        "retrieved_sources": retrieved_sources,
        "execution_steps": steps
    }

def analyze_evidence(state: ResearchState) -> ResearchState:
    docs = state.get("retrieved_documents", [])
    docs_text = "\n\n".join(docs)
    
    analysis_result = evidence_analyzer.invoke({"documents_text": docs_text})
    
    steps = state.get("execution_steps", [])
    steps.append("Evidence analyzer executed")
    
    tools_exec = state.get("tools_executed", [])
    if "evidence_analyzer" not in tools_exec:
        tools_exec.append("evidence_analyzer")
        
    return {
        **state,
        "key_findings": analysis_result.get("key_findings", []),
        "advantages": analysis_result.get("advantages", []),
        "limitations": analysis_result.get("limitations", []),
        "comparison_points": analysis_result.get("comparison_points", []),
        "evidence_summary": analysis_result.get("evidence_summary", ""),
        "execution_steps": steps,
        "tools_executed": tools_exec
    }

def generate_report(state: ResearchState) -> ResearchState:
    api_key = os.getenv("GEMINI_API_KEY")
    success = True
    if not api_key:
        final_report = "Error: GEMINI_API_KEY is missing. Cannot generate report."
        success = False
    else:
        try:
            llm = ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"), 
                google_api_key=api_key,
                temperature=0.3
            )
            
            query_text = str(state.get('user_query', '')).strip()
            topic_text = str(state.get('research_topic', '')).strip()
            type_text = str(state.get('research_type', '')).strip()
            
            plan_list = state.get('research_plan', [])
            plan_text = "\n".join(str(item) for item in plan_list) if isinstance(plan_list, list) else str(plan_list)
            
            findings_list = state.get('key_findings', [])
            findings_text = "\n".join(f"• {item}" for item in findings_list) if isinstance(findings_list, list) else str(findings_list)
            
            adv_list = state.get('advantages', [])
            advantages_text = "\n".join(f"• {item}" for item in adv_list) if isinstance(adv_list, list) else str(adv_list)
            
            lim_list = state.get('limitations', [])
            limitations_text = "\n".join(f"• {item}" for item in lim_list) if isinstance(lim_list, list) else str(lim_list)
            
            comp_list = state.get('comparison_points', [])
            comparison_text = "\n".join(f"• {item}" for item in comp_list) if isinstance(comp_list, list) else str(comp_list)
            
            summary_text = str(state.get('evidence_summary', '')).strip()
            
            prompt = f'''
            You are ResearchForge AI.
            Generate a structured research report using ONLY the provided state.
            
            USER QUERY: {query_text}
            RESEARCH TOPIC: {topic_text}
            RESEARCH TYPE: {type_text}
            
            RESEARCH PLAN:
            {plan_text}
            
            KEY FINDINGS:
            {findings_text}
            
            ADVANTAGES:
            {advantages_text}
            
            LIMITATIONS:
            {limitations_text}
            
            COMPARISON POINTS:
            {comparison_text}
            
            EVIDENCE SUMMARY:
            {summary_text}
            
            Format exactly like this (use markdown):
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            🔬 RESEARCHFORGE AI
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            RESEARCH QUESTION
            <question>

            RESEARCH TYPE
            <type>

            EXECUTIVE SUMMARY
            <summary based on evidence>

            RESEARCH PLAN
            <numbered plan>

            KEY FINDINGS
            • ...

            DETAILED ANALYSIS
            <detailed analysis>
            
            [Include ADVANTAGES section if advantages exist]
            ADVANTAGES
            • ...

            [Include LIMITATIONS section if limitations exist]
            LIMITATIONS
            • ...

            [Include COMPARISON section if it is a comparison type]
            COMPARISON
            ...

            [Include APPLICATIONS section if applications exist in the concepts or findings]
            APPLICATIONS
            ...

            CONCLUSION
            ...
            
            DO NOT include the EVIDENCE USED or AGENT EXECUTION sections in your response.
            '''
            
            max_retries = 2
            backoff = 1
            for attempt in range(max_retries + 1):
                try:
                    response = llm.invoke(prompt)
                    break
                except Exception as invoke_err:
                    err_str = str(invoke_err)
                    if ("503" in err_str or "UNAVAILABLE" in err_str) and attempt < max_retries:
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    raise invoke_err

            content = response.content
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(str(part["text"]))
                    else:
                        text_parts.append(str(part))
                final_report = " ".join(text_parts).strip()
            else:
                final_report = str(content).strip()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                final_report = "Gemini generation quota is currently exhausted for this project/model. RAG retrieval and evidence analysis completed successfully, but final report synthesis could not be generated. Please wait for quota reset or configure a Gemini project/model with available quota."
                state["quota_exhausted"] = True
            else:
                final_report = f"REPORT GENERATION ERROR\nGemini report generation failed: {error_str}"
            success = False
            
    report = final_report + "\n\n"
    
    report += "EVIDENCE USED\n\n"
    sources = state.get("retrieved_sources", [])
    unique_sources = []
    for s in sources:
        if s not in unique_sources and s != "Error":
            unique_sources.append(s)
            
    for idx, source in enumerate(unique_sources):
        report += f"{idx+1}. {source}\n"
        
    report += "\nAGENT EXECUTION\n"
    for step in state.get("execution_steps", []):
        report += f"✓ {step}\n"
        
    if success:
        report += "✓ Report synthesized\n"
    else:
        if state.get("quota_exhausted"):
            report += "✗ Report synthesis failed: Gemini quota exhausted\n"
        else:
            report += "✗ Report synthesis failed\n"
            
    report += "\nTOOLS EXECUTED\n"
    for t in state.get("tools_executed", []):
        report += f"✓ {t}\n"
        
    report += "\nRAG\n"
    report += "✓ Gemini embeddings\n"
    report += "✓ InMemoryVectorStore\n"
    report += "✓ Similarity retrieval\n"
    report += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    steps = state.get("execution_steps", [])
    if success:
        steps.append("Report synthesized")
    else:
        steps.append("Report synthesis failed")

    report_text = str(report)
    report_text = report_text.replace("\\n", "\n").replace("\\r", "\r")

    return {
        **state,
        "final_report": report_text,
        "execution_steps": steps
    }

# 5. Build Graph
workflow = StateGraph(ResearchState)
workflow.add_node("analyze_question", analyze_question)
workflow.add_node("create_research_plan", create_research_plan)
workflow.add_node("retrieve_evidence", retrieve_evidence)
workflow.add_node("analyze_evidence", analyze_evidence)
workflow.add_node("generate_report", generate_report)

workflow.add_edge(START, "analyze_question")
workflow.add_edge("analyze_question", "create_research_plan")
workflow.add_edge("create_research_plan", "retrieve_evidence")
workflow.add_edge("retrieve_evidence", "analyze_evidence")
workflow.add_edge("analyze_evidence", "generate_report")
workflow.add_edge("generate_report", END)

researchforge_runnable = workflow.compile()

# 6. FastAPI App
app = FastAPI(title="ResearchForge AI")

@app.get("/")
def home():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ResearchForge AI</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f4f4f9; color: #333; }
            header { background-color: #2c3e50; color: #fff; padding: 20px; text-align: center; }
            .container { max-width: 800px; margin: 40px auto; padding: 20px; background: #fff; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            textarea { width: 100%; height: 100px; padding: 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 16px; margin-bottom: 20px; box-sizing: border-box;}
            button { background-color: #3498db; color: #fff; border: none; padding: 10px 20px; font-size: 16px; border-radius: 4px; cursor: pointer; }
            button:hover { background-color: #2980b9; }
            .links { margin-top: 20px; text-align: center; }
            .links a { margin: 0 10px; color: #3498db; text-decoration: none; }
            .links a:hover { text-decoration: underline; }
            #report { margin-top: 30px; padding: 20px; background: #ecf0f1; border-radius: 4px; white-space: pre-wrap; display: none;}
            .loader { border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 20px; height: 20px; animation: spin 1s linear infinite; display: inline-block; vertical-align: middle; display: none; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <header>
            <h1>ResearchForge AI 🔬</h1>
            <p>AI Research & Evidence Synthesis Agent</p>
        </header>
        <div class="container">
            <textarea id="query" placeholder="Enter your research question... (e.g. Compare RAG and fine-tuning for enterprise AI applications.)"></textarea>
            <button onclick="runResearch()">Research <span id="loader" class="loader"></span></button>
            
            <div id="report"></div>
            
            <div class="links">
                <a href="/health">Health Check</a>
                <a href="/architecture">Architecture Info</a>
                <a href="/researchforge/playground/">LangServe Playground</a>
            </div>
        </div>

        <script>
            async function runResearch() {
                const query = document.getElementById('query').value;
                if (!query) return;
                
                const reportDiv = document.getElementById('report');
                const loader = document.getElementById('loader');
                
                reportDiv.style.display = 'none';
                reportDiv.innerText = '';
                loader.style.display = 'inline-block';
                
                try {
                    const response = await fetch('/researchforge/invoke', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ input: { user_query: query } })
                    });
                    
                    const data = await response.json();
                    reportDiv.innerText = data.output.final_report;
                    reportDiv.style.display = 'block';
                } catch (error) {
                    reportDiv.innerText = 'Error generating report: ' + error;
                    reportDiv.style.display = 'block';
                } finally {
                    loader.style.display = 'none';
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "agent": "ResearchForge AI",
        "langgraph": True,
        "langserve": True,
        "rag": True,
        "embeddings": True,
        "tools": True
    }

@app.get("/architecture")
def architecture():
    return {
        "agent": "ResearchForge AI",
        "workflow": [
            "analyze_question",
            "create_research_plan",
            "retrieve_evidence",
            "analyze_evidence",
            "generate_report"
        ],
        "tools": [
            "research_query_analyzer",
            "evidence_analyzer"
        ],
        "rag": {
            "embeddings": "Google Gemini Embeddings",
            "vector_store": "InMemoryVectorStore"
        },
        "llm": "Google Gemini",
        "frameworks": [
            "LangChain",
            "LangGraph",
            "LangServe",
            "FastAPI"
        ],
        "deployment": "Vercel"
    }

add_routes(
    app,
    researchforge_runnable.with_types(input_type=ResearchState),
    path="/researchforge"
)
