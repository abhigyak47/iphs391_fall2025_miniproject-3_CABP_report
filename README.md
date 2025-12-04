# 🧠 GP-Agent: Agentic-RAG for Gaussian Process Research

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange)
![RAGAS](https://img.shields.io/badge/Eval-RAGAS-red)
![Status](https://img.shields.io/badge/Status-Proposal-green)

> **Course:** IPHS 391 (Fall 2025) - Mini Project 3  
> **Author:** Abhigya Koirala  
> **Date:** November 12, 2025

---

## 📖 Abstract

**GP-Agent** is a novel **agentic-RAG system** designed for complex query-answering across dense Gaussian Process (GP) and Bayesian Optimization (BO) research literature.

Because GP research relies heavily on mathematical derivations, kernel definitions, and detailed algorithm comparisons, conventional "Naive RAG" fails to capture the necessary context. GP-Agent utilizes a **Planner–Verifier architecture** within LangGraph to dynamically select from multiple retrieval strategies, including Hierarchical RAG, Knowledge Graphs, and a custom **Kernel-Space Embedding Model**.

---

## 🚩 Problem Statement (Business Need)

**Context:** Gaussian Process literature is extremely dense, math-heavy, and cumulative. Key technical insights—such as kernel definitions and acquisition functions—are often buried in methodology sections and are difficult to retrieve via standard keyword search.

### Core Problems with Current Solutions:
1.  **High-Density + Low Discoverability:** Equations and algorithmic choices are hard to isolate.
2.  **Naive RAG Failure:** Standard RAG cannot distinguish between math vs. prose or citations vs. contributions.
3.  **Poor Synthesis:** Tools like Google Scholar cannot answer comparative or multi-hop reasoning questions (e.g., *"Compare the convergence rates of SE vs. Matern kernels"*).

### The Solution
A domain-aware autonomous research assistant capable of retrieving exact equations, comparing kernels, and grounding answers in primary literature.

---

## ⚙️ Technical Architecture

GP-Agent employs a **Planner–Verifier loop** implemented in **LangGraph**.

### 1. The Workflow
1.  **Planner Agent:** Decomposes complex user queries into structured sub-queries.
2.  **Dynamic Tool Selection:**
    * **Hierarchical RAG:** For methodology and definitions.
    * **Knowledge Graph RAG:** For mapping citations and relationships (e.g., `defines_kernel`, `extends_method`).
    * **Kernel Embedding Model (BONUS):** A fine-tuned SentenceTransformer for finding semantically similar mathematical kernels.
3.  **Synthesizer:** GPT-4o produces a draft answer.
4.  **Verifier Agent:** A Self-Reflective loop that checks grounding and math consistency using RAGAS metrics.

### 2. Component Bakeoff (Architecture Decision)

| Component | Option 1: LangGraph (Selected) | Option 2: CrewAI | Option 3: Python Script |
| :--- | :--- | :--- | :--- |
| **Strength** | Cyclical, stateful agent loops | Role-based linear workflows | Fast prototype |
| **Weakness** | Higher complexity | Poor for verification cycles | Brittle, low extensibility |
| **Why Chosen** | **Required for Planner→Verifier→Retry loops** | Not suited for multi-hop reasoning | Cannot support complex retrieval orchestration |

---

## 🛠️ Tech Stack

* **Orchestration:** LangGraph
* **LLMs:** GPT-4o (Synthesizer), Mistral-7B (Planner/Verifier)
* **Vector DB:** Weaviate / Qdrant
* **Knowledge Graph:** Neo4j
* **Parsing:** PyMuPDF, `unstructured-io`
* **Evaluation:** RAGAS, TruLens
* **Bonus Tech:** Fine-tuned Kernel-Space Embedding Model

---

## 📊 Evaluation & Benchmarking

The system is benchmarked against a **Naive RAG baseline** using a 25-question **"Gaussian Process Gold Standard"** dataset.

**Success Criteria:**
* **Faithfulness:** > 90% (measured via RAGAS)
* **Performance:** > 30% improvement over Naive RAG in Answer Relevancy and Context Precision.

---

## 🚀 Setup & Usage

### Prerequisites
* Python 3.10+
* OpenAI API Key
* Neo4j Instance (Local or AuraDB)

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/abhigyak47/iphs391_fall2025_miniproject-3_CABP_report.git](https://github.com/abhigyak47/iphs391_fall2025_miniproject-3_CABP_report.git)
   cd iphs391_fall2025_miniproject-3_CABP_report

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

3.  **Environment Setup:**
    Create a `.env` file in the root directory:
    ```env
    OPENAI_API_KEY=your_key_here
    NEO4J_URI=bolt://localhost:7687
    NEO4J_PASSWORD=your_password
    ```

### Running the Agent

```bash
python main.py --query "Compare the spectral mixture kernel to the RBF kernel regarding extrapolation."
```
-----

## 🔮 Future Work

  * **Automatic LaTeX Reconstruction:** Integrating Nougat to reconstruct true equations from PDF images.
  * **Meta-Retriever for BO Code:** Retrieving code snippets from libraries like Ax or BoTorch alongside papers.
  * **Continual Learning:** Monthly ingestion of new arXiv papers.

-----

## 📚 References

  * Group 1 Report — RAG Strategy Design
  * Group 2 Report — GraphRAG vs LightRAG
  * LangGraph Documentation
  * RAGAS & TruLens Documentation
