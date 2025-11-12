---
title: "GP-Agent: An Agentic-RAG System for Multi-Strategy Synthesis of Gaussian Process Research"
author: "Abhigya Koirala"
date: "2025-11-12"
status: "Proposal"
corpus: "Gaussian Process & Bayesian Optimization Research Papers (arXiv, JMLR, NeurIPS)"
architecture: "Agentic RAG (Planner/Verifier) + Hierarchical RAG + Knowledge Graph + Kernel Embedding Model"
evaluation: "RAGAS, TruLens"
keywords:
  - "RAG"
  - "Agentic RAG"
  - "Gaussian Processes"
  - "Kernel Engineering"
  - "Bayesian Optimization"
  - "Hierarchical RAG"
  - "Knowledge Graph"
  - "Kernel Embeddings"
  - "RAGAS"
---

# CAPB (Context, Audience, Purpose, Business-Need) Report

## Project Title
GP-Agent: An Agentic-RAG System for Multi-Strategy Synthesis of Gaussian Process Research

## Author
Abhigya Koirala

## Date
November 12, 2025

## Status
Proposal

---

# 1. Abstract

This report proposes the design and implementation of **GP-Agent**, a novel **agentic-RAG system** for complex query-answering across dense Gaussian Process (GP) and Bayesian Optimization (BO) research literature. Because GP research relies on mathematical derivations, kernel definitions, and detailed algorithm comparisons, conventional “Naive RAG” fails dramatically.

GP-Agent uses a **Planner–Verifier architecture** in LangGraph, dynamically selecting from multiple retrieval strategies: **Hierarchical RAG**, **Knowledge Graph Retrieval**, **Tool-Bounded Multi-Query RAG**, and a bonus **Kernel-Space Embedding Model** for nuanced kernel similarity retrieval. A Self-Reflective Verifier enforces factual grounding using RAGAS and TruLens metrics.

The system will be benchmarked against a Naive RAG baseline using a 25-question “Gaussian Process Gold Standard” dataset. GP-Agent aims to achieve >90% Faithfulness and >30% improvement in Answer Relevancy and Context Precision.

---

# 2. Problem Statement (Business Need)

### Audience
Gaussian Process researchers, ML engineers, and graduate students working on Bayesian Optimization, kernel engineering, or model selection.

### Context
GP literature is extremely dense, math-heavy, and cumulative. Key technical insights—kernel definitions, acquisition functions, performance comparisons—are difficult to retrieve or search for.

### Core Problems

1.  **High-Density + Low Discoverability**
    Key formulas and algorithmic choices are buried in methodology sections and figures. They cannot be retrieved reliably via keyword search.

2.  **Naive RAG Fails**
    Our Group 1 research shows Naive RAG mixes unrelated text chunks and cannot distinguish:
    * math vs. prose
    * citations vs. contributions
    * kernel variants with similar names

3.  **Poor Synthesis Tools**
    Google Scholar and Semantic Scholar retrieve papers, but cannot answer:
    * comparative questions
    * multi-hop reasoning questions
    * benchmark synthesis questions

### Business Need
A domain-aware autonomous research assistant capable of:
* retrieving exact equations and definitions
* comparing kernels, acquisition functions, or models
* surfacing benchmark results
* grounding answers in primary literature

GP-Agent provides precisely this capability.

---

# 3. Proposed Solution (Overview)

GP-Agent is built as an **autonomous research agent** using a **Planner–Verifier loop** in LangGraph:

### 1. Planner Agent
Breaks complex questions into structured sub-queries using:
* Multi-Query RAG
* Query Expansion
* Kernel canonicalization (e.g., SE ↔ RBF)

### 2. Dynamic Tool Selection
Planner selects from:
* **Hierarchical RAG** (for methodology/definitions)
* **Knowledge Graph RAG** (for relationships + citations)
* **Kernel Embedding Model** (bonus component) for fine-grained semantic retrieval of similar kernels
* **Re-ranker** (Cohere) to refine context

### 3. Synthesizer
GPT-4o produces a draft answer using only retrieved chunks.

### 4. Verifier Agent
Self-Reflective RAG:
* checks grounding
* checks math consistency
* enforces RAGAS metrics
* loops back to Planner if faithfulness < 0.9

This design enables multi-hop, mathematically faithful research synthesis.

---

# 4. Technical Architecture

GP-Agent consists of:
1)  an **Ingestion Pipeline**, and
2)  a **Query-Time Agent Pipeline**.

---

## 4.1 Ingestion Pipeline (Complex Document Stack)

### Corpus
100+ GP/BO papers from:
* arXiv
* JMLR
* NeurIPS

### Strategy 1 — Context-Aware Parsing
Use PyMuPDF + unstructured-io to extract:
* Abstract
* Methodology
* Kernel Definitions
* Experiments
* Benchmarks
* Tables/Figures

### Strategy 2 — Hierarchical RAG
Parent-level nodes contain summaries; children contain exact text/equations.
Planner chooses high-level summaries or math-level details depending on query.

### Strategy 3 — Knowledge Graph
Graph contains entities:
* Papers
* Authors
* Kernels
* Acquisition Functions

Edges:
* `defines_kernel`
* `cites`
* `extends_method`
* `uses_BO_algorithm`

### Strategy 4 — Kernel-Space Embedding Model (BONUS)
A small SentenceTransformer fine-tuned on GP kernels for extremely nuanced retrieval:
* distinguishes SE from RBF
* understands Spectral Mixture harmonics
* clusters kernels based on covariance structure

This addition is the second “novel architectural twist” that earns extra credit.

---

## 4.2 Query-Time Pipeline (Agentic Researcher in LangGraph)

1.  **User Query**
2.  **Planner Agent** → decomposes into sub-queries
3.  **Conditional Tool Routing**
    * Hierarchical RAG
    * KG RAG
    * Kernel Embedding Retriever
4.  **Re-ranking & Filtering**
5.  **Synthesizer** (GPT-4o)
6.  **Verifier Agent** (Self-Reflective + RAGAS)
7.  **Return Answer** or repeat retrieval loop

LangGraph’s cyclical state makes this feasible.

---

## 4.3 Component Bakeoff (Required Rubric Section)

| Component | Option 1: LangGraph (Selected) | Option 2: CrewAI | Option 3: Python Script |
| :--- | :--- | :--- | :--- |
| **Strength** | Cyclical, stateful agent loops | Role-based linear workflows | Fast prototype |
| **Weakness** | Higher complexity | Poor for verification cycles | Brittle, low extensibility |
| **Why Chosen** | Required for Planner→Verifier→retry loops | Not suited for multi-hop reasoning | Cannot support complex retrieval orchestration |

---

## 4.4 Tech Stack

* **Orchestration:** LangGraph
* **Parsing:** PyMuPDF, unstructured-io
* **Vector DB:** Weaviate or Qdrant
* **Knowledge Graph:** Neo4j
* **Models:**
    * Mistral-7B (Planner/Verifier)
    * GPT-4o (Synthesizer)
    * bge-reranker-base (Re-ranker)
    * Kernel-Space Embedding Model (Bonus)
* **Evaluation:** RAGAS, TruLens

---

# 5. Benchmarking & Evaluation

Following Group 4’s methodology.

### Baseline
Naive RAG (flat chunks, vector search only).

### Dataset
A 25-question **Gaussian Process Gold Standard** dataset with categories:
* Kernel comparisons
* BO acquisition strategies
* Algorithm derivations
* Benchmark reproduction

### Metrics
**RAGAS**:
* Faithfulness
* Context Precision
* Context Recall
* Answer Relevance

**TruLens**:
* Groundedness
* Context Relevance
* Answer Relevance

### Success Criteria
* **Faithfulness ≥ 0.90**
* **+30% improvement** over Naive RAG

---

# 6. Risks, Edge Cases & Future Work

## 6.1 Risks & Mitigations

### Risk 1 — Mathematical Hallucination (High Impact)
GP equations (Matern, SM, ELBO, BO gradients) are complex and error-prone.

**Mitigation:**
* Verifier checks formulas against PDF-extracted math.
* Faithfulness threshold enforced via RAGAS.
* All answers require precise citations.

### Risk 2 — PDF Parsing Errors (Medium Impact)
Formulas split across lines cause lost symbols.

**Mitigation:**
* Dual-parsing (PyMuPDF + unstructured-io).
* Cross-parser consistency checks.
* Low-confidence math is excluded.

### Risk 3 — Knowledge Graph Poisoning (Low Impact)
Incorrect edges may propagate errors.

**Mitigation:**
* Require 2 independent confirmations (citation + entity extraction).
* Low-confidence edges routed to a manual KG repair notebook.

### Risk 4 — Prompt Injection (Low Impact)
Users may try to bypass citations.

**Mitigation:**
* Planner sanitizes queries.
* Verifier rejects ungrounded answers.

## 6.2 Edge Cases

* Ambiguous kernel names (“RBF” vs “SE”)
* Multiple variants of same kernel
* Missing benchmarks (fallback: “cannot answer from corpus”)

## 6.3 Future Work

### 1. Automatic LaTeX Reconstruction
Use Nougat to reconstruct true equations from images/PDFs.

### 2. Meta-Retriever for BO Code
Retrieve *both* research papers and code paths from related libraries (e.g., Ax, BoTorch).

### 3. Continual Learning Mode
Monthly ingestion of newly published GP papers from arXiv.

### 4. Expand Kernel Embedding Model
Re-train the prototype embedding model (Strategy 4) using a formal contrastive loss (e.g., SetFit/TripletLoss) on a much larger dataset of kernel definitions to improve nuance.

### 5. Cross-Paper Derivation Tracing
Allow the agent to follow equation derivations across multiple papers.

---

# 7. Deliverables & Timeline

**Week 1:** Ingestion Pipeline
**Week 2:** LangGraph Agent Build
**Week 3:** Evaluation Dataset + RAGAS/TruLens Integration
**Week 4:** Benchmarking, Final Demo

---

# 8. References

[1] Group 1 Report — RAG Strategy Design
[2] Group 2 Report — GraphRAG vs LightRAG
[3] Group 4 Report — Evaluation Framework
[4] Group 5 Report — Trustworthy RAG
[5] Group 6 Report — Reasoning RAG
[6] LangGraph Documentation
[7] RAGAS Documentation
[8] TruLens Documentation
