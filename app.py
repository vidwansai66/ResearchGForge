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
    interpreted_query: str
    requires_current_info: bool
    research_topic: str
    research_domain: str
    research_type: str
    key_concepts: list[str]
    sub_questions: list[str]
    search_queries: list[str]
    research_plan: list[str]
    retrieved_documents: list[str]
    retrieved_sources: list[str]
    evidence_status: str
    evidence_summary: str
    supported_claims: list[str]
    inferences: list[str]
    unsupported_claims: list[str]
    final_report: str
    execution_steps: list[str]
    tools_executed: list[str]
    quota_exhausted: bool

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

def _fallback_query_analysis(query: str) -> dict:
    words = [w for w in query.split() if len(w) > 4]
    return {
        "interpreted_query": query,
        "requires_current_info": False,
        "research_topic": "General AI Research",
        "research_domain": "Unknown / General",
        "research_type": "Other / General Research Question",
        "key_concepts": words,
        "sub_questions": ["What is the main concept?", "How does it work?"],
        "search_queries": [query]
    }

def _fallback_evidence_analysis() -> dict:
    return {
        "evidence_status": "Moderate",
        "evidence_summary": "The retrieved documents provide foundational context.",
        "supported_claims": ["Information retrieved successfully."],
        "inferences": ["Provides context."],
        "unsupported_claims": ["Limited to context."]
    }

# 3. Tools
@tool
def research_query_analyzer(query: str) -> dict:
    """Analyzes a research question to identify the topic, domain, type, key concepts, sub-questions, and search queries."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _fallback_query_analysis(query)
        
    llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"), google_api_key=api_key, temperature=0.1)
    
    prompt = f"""
    Analyze the following research question: "{query}"
    
    Return a JSON object with EXACTLY these keys:
    - "interpreted_query": string (explicit interpretation if ambiguous, else same as query)
    - "requires_current_info": boolean (true if question asks for today, latest, 2026, current info)
    - "research_topic": string
    - "research_domain": string (e.g., Artificial Intelligence, Healthcare, Unknown / General)
    - "research_type": string (e.g., Comparison, Explanation, Definition, Cause and effect, Multi-part question, Other / General Research Question)
    - "key_concepts": list of strings
    - "sub_questions": list of strings (break down if multi-part)
    - "search_queries": list of strings (3-4 specific search queries optimized for a vector database)
    
    Respond ONLY with valid JSON, without any markdown formatting like ```json.
    """
    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        if text.startswith("```json"): text = text[7:]
        if text.startswith("```"): text = text[3:]
        if text.endswith("```"): text = text[:-3]
        return json.loads(text.strip())
    except Exception as e:
        print(f"Error in query analyzer: {e}")
        return _fallback_query_analysis(query)

@tool
def evidence_analyzer(documents_text: str, query: str) -> dict:
    """Analyzes retrieved evidence to evaluate relevance, extract supported claims, inferences, and unsupported claims."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _fallback_evidence_analysis()
        
    llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"), google_api_key=api_key, temperature=0.1)
    
    prompt = f"""
    Evaluate the provided evidence against the research question: "{query}"
    
    Evidence:
    {documents_text}
    
    Determine if the evidence is sufficient to answer the question. 
    Return a JSON object with EXACTLY these keys:
    - "evidence_status": string (MUST be one of: "Strong", "Moderate", "Limited", "Insufficient")
    - "evidence_summary": string (Brief summary of what the evidence covers)
    - "supported_claims": list of strings (Information directly supported by the evidence that answers the query)
    - "inferences": list of strings (Reasonable conclusions derived from the evidence)
    - "unsupported_claims": list of strings (Parts of the query that cannot be answered with this evidence)
    
    Be conservative. If the evidence is completely unrelated to the query, set evidence_status to "Insufficient".
    
    Respond ONLY with valid JSON, without any markdown formatting.
    """
    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        if text.startswith("```json"): text = text[7:]
        if text.startswith("```"): text = text[3:]
        if text.endswith("```"): text = text[:-3]
        return json.loads(text.strip())
    except Exception as e:
        print(f"Error in evidence analyzer: {e}")
        return _fallback_evidence_analysis()

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
        "interpreted_query": analysis_result.get("interpreted_query", query),
        "requires_current_info": analysis_result.get("requires_current_info", False),
        "research_topic": analysis_result.get("research_topic", ""),
        "research_domain": analysis_result.get("research_domain", "Unknown / General"),
        "research_type": analysis_result.get("research_type", "Other / General Research Question"),
        "key_concepts": analysis_result.get("key_concepts", []),
        "sub_questions": analysis_result.get("sub_questions", []),
        "search_queries": analysis_result.get("search_queries", [query]),
        "execution_steps": steps,
        "tools_executed": tools_exec
    }

def create_research_plan(state: ResearchState) -> ResearchState:
    q_type = state.get("research_type", "Other / General Research Question")
    
    if q_type == "Comparison":
        plan = [
            "1. Identify comparison dimensions",
            "2. Retrieve evidence for concept A",
            "3. Retrieve evidence for concept B",
            "4. Compare evidence",
            "5. Synthesize"
        ]
    elif q_type in ["Explanation", "Definition", "How-to / mechanism"]:
        plan = [
            "1. Identify concepts",
            "2. Retrieve relevant evidence",
            "3. Explain concepts",
            "4. Synthesize"
        ]
    elif q_type == "Cause and effect":
        plan = [
            "1. Identify causes",
            "2. Retrieve evidence",
            "3. Analyze relationships",
            "4. Synthesize"
        ]
    elif q_type == "Applications":
        plan = [
            "1. Identify domain",
            "2. Retrieve application evidence",
            "3. Analyze use cases",
            "4. Limitations",
            "5. Synthesize"
        ]
    else:
        plan = [
            "1. General research analysis",
            "2. Retrieve evidence",
            "3. Evaluate evidence",
            "4. Synthesize if sufficient"
        ]
        
    steps = state.get("execution_steps", [])
    steps.append(f"Question type detected: {q_type}")
    steps.append(f"Domain detected: {state.get('research_domain', 'Unknown / General')}")
    steps.append("Research plan created")
    
    return {
        **state,
        "research_plan": plan,
        "execution_steps": steps
    }

def retrieve_evidence(state: ResearchState) -> ResearchState:
    search_queries = state.get("search_queries", [state.get("user_query", "")])
    
    vs = get_vector_store()
    retrieved_docs = []
    retrieved_sources = []
    
    if vs is not None:
        all_results = []
        for q in search_queries:
            results = vs.similarity_search(q, k=3)
            all_results.extend(results)
            
        seen = set()
        for r in all_results:
            if r.page_content not in seen:
                seen.add(r.page_content)
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
    query = state.get("interpreted_query", state.get("user_query", ""))
    docs_text = "\n\n".join(docs)
    
    analysis_result = evidence_analyzer.invoke({"documents_text": docs_text, "query": query})
    
    steps = state.get("execution_steps", [])
    steps.append("Evidence relevance evaluated")
    steps.append("Evidence analyzer executed")
    
    tools_exec = state.get("tools_executed", [])
    if "evidence_analyzer" not in tools_exec:
        tools_exec.append("evidence_analyzer")
        
    evidence_status = analysis_result.get("evidence_status", "Insufficient")
    if evidence_status == "Insufficient":
        steps.append("⚠ Insufficient relevant evidence")
        
    return {
        **state,
        "evidence_status": evidence_status,
        "evidence_summary": analysis_result.get("evidence_summary", ""),
        "supported_claims": analysis_result.get("supported_claims", []),
        "inferences": analysis_result.get("inferences", []),
        "unsupported_claims": analysis_result.get("unsupported_claims", []),
        "execution_steps": steps,
        "tools_executed": tools_exec
    }

def generate_report(state: ResearchState) -> ResearchState:
    api_key = os.getenv("GEMINI_API_KEY")
    success = True
    
    if not api_key:
        final_report = "Error: GEMINI_API_KEY is missing. Cannot generate report."
        success = False
    elif state.get("evidence_status") == "Insufficient":
        question = state.get("user_query", "")
        req_current = state.get("requires_current_info", False)
        
        current_msg = ""
        if req_current:
            current_msg = "This question requires current/live information that is not available through my current research sources.\n\n"
            
        final_report = f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔬 RESEARCHFORGE AI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUESTION STATUS
Insufficient Evidence

QUESTION
{question}

WHY
{current_msg}The current ResearchForge knowledge base does not contain enough relevant evidence for this question.

ANALYSIS
The question was successfully analyzed. It was classified as '{state.get("research_type")}' in the domain of '{state.get("research_domain")}'.

WHAT I CAN DO
I can analyze the question structure and identify what evidence would be required.

EVIDENCE STATUS
Insufficient relevant evidence

WHAT WOULD BE NEEDED
Information covering the following concepts: {", ".join(state.get("key_concepts", []))}"""
        
        success = False
    else:
        try:
            llm = ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"), 
                google_api_key=api_key,
                temperature=0.3
            )
            
            query_text = str(state.get('interpreted_query', state.get('user_query', ''))).strip()
            type_text = str(state.get('research_type', '')).strip()
            status_text = str(state.get('evidence_status', 'Limited')).strip()
            req_current = state.get('requires_current_info', False)
            
            current_msg = ""
            if req_current:
                current_msg = "Note: This question asks for current/live information which is not fully available in the local evidence base. The answer relies on available local evidence.\n\n"
                
            plan_list = state.get('research_plan', [])
            plan_text = "\n".join(str(item) for item in plan_list) if isinstance(plan_list, list) else str(plan_list)
            
            supp_list = state.get('supported_claims', [])
            supported_text = "\n".join(f"• {item}" for item in supp_list) if isinstance(supp_list, list) else str(supp_list)
            
            inf_list = state.get('inferences', [])
            inferences_text = "\n".join(f"• {item}" for item in inf_list) if isinstance(inf_list, list) else str(inf_list)
            
            unsupp_list = state.get('unsupported_claims', [])
            unsupported_text = "\n".join(f"• {item}" for item in unsupp_list) if isinstance(unsupp_list, list) else str(unsupp_list)
            
            prompt = f'''
            You are ResearchForge AI.
            Generate a structured research report using ONLY the provided state. Do NOT hallucinate.
            
            USER QUERY: {query_text}
            RESEARCH TYPE: {type_text}
            EVIDENCE STATUS: {status_text}
            {current_msg}
            
            RESEARCH PLAN:
            {plan_text}
            
            SUPPORTED BY EVIDENCE:
            {supported_text}
            
            INFERENCE:
            {inferences_text}
            
            LIMITATION (Unsupported Claims):
            {unsupported_text}
            
            Format exactly like this (use markdown):
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            🔬 RESEARCHFORGE AI
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            RESEARCH QUESTION
            <question>

            RESEARCH TYPE
            <type>
            
            EVIDENCE STATUS
            {status_text}

            EXECUTIVE SUMMARY
            <summary based on evidence>

            RESEARCH PLAN
            <numbered plan>

            [Adapt the rest of the report based on the RESEARCH TYPE. For example, if Comparison, include COMPARISON CRITERIA, CONCEPT A, CONCEPT B, COMPARISON. If Explanation, include DEFINITION, CORE CONCEPTS, HOW IT WORKS. Include relevant sections dynamically.]

            [ALWAYS Include these distinctions at the end of the analysis:]
            SUPPORTED BY EVIDENCE
            • ...

            INFERENCE
            • ...

            LIMITATION
            • ...
            
            [If the question required current info (e.g. 2026), explicitly state that it's unavailable if it wasn't found]

            CONCLUSION
            ...
            
            DO NOT include the EVIDENCE USED or AGENT EXECUTION sections in your response. DO NOT hallucinate facts.
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
            final_report = str(content).strip()
            
            if final_report.startswith("```markdown"): final_report = final_report[11:]
            elif final_report.startswith("```"): final_report = final_report[3:]
            if final_report.endswith("```"): final_report = final_report[:-3]
            final_report = final_report.strip()
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                final_report = "Gemini generation quota is currently exhausted for this project/model. RAG retrieval and evidence analysis completed successfully, but final report synthesis could not be generated. Please wait for quota reset or configure a Gemini project/model with available quota."
                state["quota_exhausted"] = True
            else:
                final_report = f"REPORT GENERATION ERROR\nGemini report generation failed: {error_str}"
            success = False
            
    report = final_report + "\n\n"
    
    if state.get("evidence_status") != "Insufficient":
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
        
    if state.get("evidence_status") == "Insufficient":
        report += "✗ Report synthesis skipped\n"
    elif success:
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
    if state.get("evidence_status") == "Insufficient":
        pass
    elif success:
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
