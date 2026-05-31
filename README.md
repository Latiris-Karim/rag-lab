# rag-lab

A personal lab for experimenting with different RAG architectures. Each architecture is self-contained and built on shared base classes.

## Goal

Understand the tradeoffs between RAG architectures by building and running retrieval tests

## Architectures

| Architecture | Status | Description |
|---|---|---|
| Standard RAG | Done | Basic chunking + embeddings + vector retrieval |
| Contextual RAG | Done | LLM-generated context prepended to each chunk + chunk |
| Hybrid RAG | Done | Contextual Chunks + BM25 indexing + reranker |

## Stack

- **Chunking** — [Chonkie](https://github.com/chonkie-ai/chonkie) (`RecursiveChunker` with markdown recipe)
- **Embeddings** — `sentence-transformers/all-MiniLM-L6-v2`
- **Vector DB** — ChromaDB
- **LLM** — DeepSeek via OpenAI-compatible API
- **Document ingestion** — [MarkItDown](https://github.com/microsoft/markitdown) (converts non-markdown files)
- **Reranker Model** — `BAAI/bge-reranker-large`



## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file or copy `.env.example` to .env and fill in the variables:
```
API_KEY=your_key_here
rag_files=path/to/your/documents
db_path=path/to/db/
```

## Comparing RAGs 
Running the RAGs on my testset with top k =  3  the following % represent how often it got the "golden chunk" for that question. 


### Results:

Hybrid:     87.3%

Contextual: 82.5%

Standard:   48.4%


## Conclusion

Standard to Contextual +34% accuracy increase.

 LLM-generated context summaries give embeddings a much clearer signal about what each chunk is about.

Contextual to Hybrid +5% accuracy increase.

 BM25 catches keyword-heavy queries that semantic search misses, with the reranker reconciling both signals.

### Side Note

The test was run with top k = 3 purposely, increasing this will naturally improve the retrieval at the cost of tokens.

You're welcome to run your own test variations
