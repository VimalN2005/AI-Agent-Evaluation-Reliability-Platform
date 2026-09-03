 # 🤖 AI Agent Evaluation & Reliability Platform

[![CI](https://github.com/VimalN2005/AI-Agent-Evaluation-Reliability-Platform/actions/workflows/ci.yml/badge.svg)](https://github.com/VimalN2005/AI-Agent-Evaluation-Reliability-Platform/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![GitHub Repo stars](https://img.shields.io/github/stars/VimalN2005/AI-Agent-Evaluation-Reliability-Platform?style=social)](https://github.com/VimalN2005/AI-Agent-Evaluation-Reliability-Platform/stargazers)

> **Production-grade platform to evaluate, monitor, compare, and benchmark AI Agents & GenAI applications** — detecting hallucinations, measuring RAG retrieval accuracy, validating tool calls, and tracking token cost & latency.

---

## 📌 Key Evaluation Pillars

| Pillar | Metric | Description |
|---|---|---|
| 🎯 **Correctness** | LLM-as-Judge & Semantic Match | Compares agent outputs against ground truth / expected answers. |
| 🛡️ **Faithfulness** | Hallucination Detection | Verifies whether the generated answer is strictly grounded in retrieved context. |
| 🔍 **RAG Quality** | Context Precision & Recall | Measures relevance of retrieved chunks and documents. |
| ⚡ **Tool-Call Accuracy** | Function Calling Validation | Evaluates whether the agent invoked the correct tool with expected arguments. |
| 💰 **Cost & Latency** | Token Usage & Runtime (ms) | Real-time monitoring of tokens, API cost ($), and response latency. |
| 🚨 **Safety Guardrails** | Rule-based & Prompt Safety | Automatically flags unsafe, abusive, or out-of-boundary outputs. |

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A[User / Client Request] --> B[AI Agent Under Test]
    B --> C[SDK / Trace Collector\nsdk/client.py]
    C --> D[(SQLite Trace DB\ndb.py)]
    D --> E[Evaluation Engine\nengine.py]
    E --> F1[Correctness Evaluator]
    E --> F2[Faithfulness Evaluator]
    E --> F3[RAG Precision & Recall]
    E --> F4[Tool-Call Accuracy]
    E --> F5[Cost & Latency Analyzer]
    F1 & F2 & F3 & F4 & F5 --> G[(Evaluations DB)]
    G --> H[Streamlit Analytics Dashboard\napp.py]
    H --> I[Compare Runs & Regression Reports]
