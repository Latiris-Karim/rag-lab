# rag-lab

A personal lab for experimenting with different RAG architectures. Each architecture is self-contained and built on shared base classes.

## Goal

Understand the tradeoffs between RAG architectures by building and running retrieval tests

## Architectures

| Architecture | Status | Description |
|---|---|---|
| Standard RAG | Done | Basic chunking + embeddings + vector retrieval |
| Contextual RAG | Done | LLM-generated context prepended to each chunk |
| Hybrid RAG | Planned | BM25 + reranker |

## Stack

- **Chunking** — [Chonkie](https://github.com/chonkie-ai/chonkie) (`RecursiveChunker` with markdown recipe)
- **Embeddings** — `sentence-transformers/all-MiniLM-L6-v2`
- **Vector DB** — ChromaDB
- **LLM** — DeepSeek via OpenAI-compatible API
- **Document ingestion** — [MarkItDown](https://github.com/microsoft/markitdown) (converts non-markdown files)



## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
API_KEY=your_key_here
rag_files=path/to/your/documents
db_path=path/to/db/
```

## Status

Work in progress. Architectures are added and evaluated incrementally — an `eval.py` for comparing architectures is planned.

## Comparing RAGs 

//PLANNED
