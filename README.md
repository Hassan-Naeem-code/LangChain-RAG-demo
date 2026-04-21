# LangChain RAG Demo

A minimal end-to-end retrieval-augmented generation (RAG) pipeline in a single Python script. Loads a text file, chunks it, embeds the chunks, stores them in Chroma, and answers a query using an OpenAI LLM grounded in the retrieved context.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1c3c3c)
![Chroma](https://img.shields.io/badge/Chroma-vector_store-ff6f61)
![License](https://img.shields.io/badge/License-MIT-green)

## What you'll learn

- The four core stages of a RAG pipeline — **load → split → embed → retrieve → generate**
- Where to plug in different loaders, splitters, vector stores, and LLMs
- The smallest possible working setup you can build on

## Pipeline

```
sample_docs.txt
      │
      ▼
TextLoader ── CharacterTextSplitter(chunk_size=500, overlap=50)
      │
      ▼
OpenAIEmbeddings ──► Chroma (in-memory vector store)
      │
      ▼
RetrievalQA(llm=OpenAI, chain_type="stuff")
      │
      ▼
query → retrieved chunks + query → LLM → Answer
```

## Quick start

```bash
git clone https://github.com/Hassan-Naeem-code/LangChain-RAG-demo.git
cd LangChain-RAG-demo

python -m venv .venv && source .venv/bin/activate
pip install langchain langchain-community openai chromadb tiktoken python-dotenv

# Add your OpenAI key
echo "OPENAI_API_KEY=sk-..." > .env

python rag_demo.py
```

Output:

```
Answer: LangChain is a framework for developing applications powered by language models...
```

## What the script does (step by step)

1. **Load** — reads `sample_docs.txt` with `TextLoader`. Swap for `PyPDFLoader`, `WebBaseLoader`, or `DirectoryLoader` for other sources.
2. **Split** — breaks the text into 500-char chunks with 50-char overlap. For production RAG consider `RecursiveCharacterTextSplitter` or structural splitting (see [RAG chunking strategies](https://gist.github.com/Hassan-Naeem-code/6906113520019939081256598012b3ed)).
3. **Embed + store** — uses OpenAI embeddings and an ephemeral Chroma store. For persistent storage add `persist_directory="./chroma_db"`.
4. **Retrieve** — `db.as_retriever()` returns the top matches for the query (default k=4).
5. **Generate** — the `"stuff"` chain concatenates retrieved chunks into the prompt. For longer contexts switch to `"map_reduce"` or `"refine"`.

## Customize

| Want to… | Change |
|---|---|
| Use a different LLM | `OpenAI(...)` → `ChatOpenAI(model="gpt-4o")` or `ChatGoogleGenerativeAI(...)` |
| Persist the vector store | `Chroma.from_documents(..., persist_directory="./chroma_db")` |
| Load PDFs | `from langchain.document_loaders import PyPDFLoader` |
| Use open embeddings | `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` |
| Tune retrieval | `db.as_retriever(search_kwargs={"k": 6})` |

## Files

```
.
├── rag_demo.py         # full pipeline in ~40 lines
├── sample_docs.txt     # the document corpus
└── README.md
```

## Next steps

- Swap `TextLoader` for a directory of real docs
- Add a rerank step with a cross-encoder for better precision
- Move to prompt caching + small local embeddings to cut cost
- See a framework-free version: [minimal_rag.py gist](https://gist.github.com/Hassan-Naeem-code/f22285f7d5d590e4bfd233058fddfbc7)

## License

MIT
