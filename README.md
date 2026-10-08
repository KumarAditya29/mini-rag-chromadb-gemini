# Mini RAG — ChromaDB + Gemini

A minimal Retrieval-Augmented Generation (RAG) system built with Python.

## Architecture
Documents → Chunking → ChromaDB embeddings → Top-3 retrieval → Gemini → Answer + Sources

## Features
- Ingests local PDF/text documents
- ~500-word chunks with light overlap
- Persistent local ChromaDB vector store
- Local default Chroma embedding function
- Gemini for grounded answer generation
- CLI question-answering loop
- Displays retrieved source pages/chunks
- Refuses to answer when information is not in the knowledge base

## Run

```bash
python ingest.py
python query.py
