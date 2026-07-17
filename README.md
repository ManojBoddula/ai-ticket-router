# Semantic Support Ticket Router

A production-grade, local AI pipeline built using LangChain and ChromaDB designed to automatically categorize incoming support tickets, resolve common FAQs via semantic matching, and securely route unique edge cases to human agents.

## 🛠️ Architecture Overview

Instead of relying on fragile, keyword-based filtering (which falls apart with typos or varying phrasing), this system processes inputs via **Semantic Text Clustering**:

1. **Vector Space Mapping:** Unstructured ticket text is converted into 384-dimensional dense vectors using a local sentence-transformer embedding model (`all-MiniLM-L6-v2`).
2. **Intent Grouping:** 
   * **Cluster A (Authentication & Access):** High semantic similarity links queries like *"I forgot my password"* and *"I can't log in, as password is incorrect"* together, identifying them as a single structural intent (Account Access).
   * **Cluster B (HR Operations):** Queries like *"How to see leave balance?"* map to a completely separate geometric vector space handling internal employee database lookups.
3. **Anti-Hallucination Guardrail:** The system evaluates similarity scores. If a match falls below the confidence threshold slider, the code dynamically overrides auto-resolution and alters the payload action to `ROUTE_TO_HUMAN`, completely preventing the AI from guessing or delivering wrong answers.

## 🚀 Quick Start

### Prerequisites
Install the modernized 2026 LangChain core ecosystem packages:
```bash
pip install langchain-chroma langchain-huggingface