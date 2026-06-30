# LocalDoc GraphRAG

A private, fully local, multi-document question-answering application built with **Streamlit, FastAPI, Ollama, FastEmbed, LangGraph, and Neo4j Community Edition**.

LocalDoc GraphRAG lets you place PDF files in a local folder, select one or more documents, and ask grounded questions about their contents. The project runs without paid API keys and keeps document processing on your machine.

## Features

- Fully local document question answering
- Multi-PDF input
- Streamlit user interface
- FastAPI backend
- Local LLM inference through Ollama
- Semantic and keyword retrieval
- Neo4j-backed document and graph storage
- Automatic detection of new, changed, and removed files
- Persistent indexing for unchanged documents
- Document-level filtering
- Evidence-based citations
- Clean explanatory and compliance-style answers
- Two response modes:
  - **Instant** — skips LLM synthesis and returns evidence directly from retrieved document text
  - **Balanced** — uses Ollama to generate a more polished response
- Extractive fallback when Ollama times out
- No paid API keys required

## How It Works

1. Add PDF files to the `data/` folder.
2. Start the application.
3. Select one or more documents in Streamlit.
4. Ask a question.
5. The selected files are parsed, chunked, embedded, and stored in Neo4j.
6. The retriever finds relevant document sections.
7. The application returns a grounded response with citations.

Files that have already been indexed are checked using their content hash and are skipped when unchanged.

## Requirements

Install the following tools before running the project.

### Required

- **Python 3.11 or 3.12**
- **Docker Desktop**
- **Ollama**
- **Git** for cloning the repository

### Recommended

- At least **8 GB RAM**
- Apple Silicon or another modern CPU
- Sufficient disk space for Docker, Neo4j data, Ollama models, embedding models, and indexed documents

## Installation

Clone the repository and open the project folder:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd free-open-source-agentic-graphrag
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create the local environment file:

```bash
cp .env.example .env
```

Download the Ollama model:

```bash
ollama pull qwen3:1.7b
```

Make sure Docker Desktop is running before starting the application.

## Add Input Documents

Place PDF files inside:

```text
data/
```

Example:

```text
data/
├── document_1.pdf
├── document_2.pdf
└── document_3.pdf
```

The application reads the files directly from this folder.

### Changing the Input Data

To use a different document collection:

1. Stop the application.
2. Remove the old PDF files from `data/`.
3. Copy the new PDF files into `data/`.
4. Start the application again.

The Streamlit document selector will reflect the current contents of the `data/` folder.

When a document is:

- **New** — it is indexed when selected.
- **Modified** — its stored index is rebuilt.
- **Unchanged** — indexing is skipped.
- **Deleted** — stale indexed data is removed during the next query workflow.

## Run the Application

The recommended command is:

```bash
./.venv/bin/python scripts/run_ui.py
```

You can also use:

```bash
./run.sh
```

The launcher:

1. Checks Docker and Neo4j.
2. Checks Ollama availability.
3. Verifies the configured Ollama model.
4. Starts the FastAPI backend.
5. Waits for backend health.
6. Starts Streamlit.

Open:

```text
http://localhost:8501
```

## Services

| Service | Address |
|---|---|
| Streamlit UI | `http://localhost:8501` |
| FastAPI backend | `http://127.0.0.1:8000` |
| FastAPI documentation | `http://127.0.0.1:8000/docs` |
| Neo4j Browser | `http://localhost:7474` |
| Ollama | `http://localhost:11434` |

## Answer Modes

### Instant

Instant mode avoids Ollama answer generation and returns grounded statements directly from the retrieved document text.

Use it when:

- You want the fastest response.
- You are testing retrieval.
- Your computer has limited memory.
- Ollama generation is timing out.

### Balanced

Balanced mode sends retrieved context to Ollama and produces a more natural, organized response.

Use it when:

- You want a polished explanation.
- The selected documents are small or medium-sized.
- Your computer has enough memory.
- A slightly longer response time is acceptable.

## Large PDF and Performance Limitations

Large PDFs can take significant time during their first ingestion.

The application may need to:

- Extract text from every page
- Split the document into chunks
- Generate embeddings
- Identify entities and relationships
- Store data in Neo4j
- Build or update retrieval indexes

A large regulation, manual, textbook, or report may therefore take several minutes to index.

For better performance:

- Start with one small PDF.
- Select only the documents needed for the current question.
- Add large documents one at a time.
- Use **Instant** mode.
- Avoid selecting every document unless the question requires all of them.
- Keep Ollama and Docker running before starting the app.
- Use a smaller local model on low-memory computers.

### Possible Large-File Errors

Very large or complex files may fail because of:

- Insufficient system memory
- A local Ollama model that is too small for the requested context
- Ollama request timeouts
- Very long document context
- Image-heavy, scanned, corrupted, or poorly encoded PDFs
- Docker or Neo4j resource limits
- Too many selected documents

When this happens:

1. Stop the application.
2. Test with one small PDF.
3. Use **Instant** mode.
4. Restart Ollama and Docker.
5. Reduce the number of selected documents.
6. Split a very large PDF into smaller files.
7. Use a more capable Ollama model when your hardware supports it.

The application includes an extractive fallback for some Ollama timeout cases, but local hardware limits can still affect ingestion and answer generation.

## Recommended Local Settings

The `.env.example` contains MacBook-friendly defaults similar to:

```dotenv
FAST_MODE=true
REQUEST_TIMEOUT_SECONDS=180
GENERATION_MODEL=qwen3:1.7b
OLLAMA_NUM_CTX=4096
MAX_OUTPUT_TOKENS=450
VECTOR_TOP_K=8
KEYWORD_TOP_K=10
RERANK_TOP_K=6
RERANK_CANDIDATE_K=24
ANSWER_CONTEXT_CHARS=10000
```

These settings reduce memory use and response time. Larger models and context windows may improve answer quality but require more RAM and processing time.

## Troubleshooting

### Streamlit shows old documents

Confirm that only the desired PDF files remain in `data/`, then restart the application.

If necessary, remove the UI cache:

```bash
rm -f .cache/streamlit_startup_cache.pkl
```

### `ModuleNotFoundError: enterprise_graphrag`

Use the launcher:

```bash
./.venv/bin/python scripts/run_ui.py
```

Do not start `streamlit_app.py` directly unless `PYTHONPATH` includes the project’s `src/` folder.

### Ollama timeout or server disconnected

Check Ollama:

```bash
ollama list
ollama ps
```

Test the configured model:

```bash
ollama run qwen3:1.7b "Reply only with OK"
```

Then restart the application.

### Neo4j is unavailable

Make sure Docker Desktop is open, then run:

```bash
docker compose up -d
```

### First query is slow

The first query on a new document may trigger parsing, chunking, embeddings, graph creation, and database storage.

Later queries on the same unchanged document should be faster.

## Stop the Application

Press:

```text
Control + C
```

Neo4j data remains persistent in Docker.

To stop Neo4j:

```bash
docker compose down
```

Do not run:

```bash
docker compose down -v
```

unless you intentionally want to erase the entire Neo4j database.

## Project Structure

```text
free-open-source-agentic-graphrag/
├── data/                     # Input PDF files
├── scripts/                  # Startup and configuration scripts
├── src/
│   └── enterprise_graphrag/  # Main application package
├── artifacts/                # Generated local artifacts
├── streamlit_app.py          # Streamlit interface
├── docker-compose.yml        # Neo4j service configuration
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project configuration
├── run.sh                    # One-command launcher
└── README.md
```

## GitHub Safety

Do not commit private, confidential, copyrighted, or employer-owned documents.

The repository should exclude:

```text
.env
.venv/
.cache/
.model_cache/
artifacts/
data/*.pdf
__pycache__/
*.pyc
.DS_Store
```

Keep `data/.gitkeep` so the empty folder remains in Git.

For public demonstrations, use documents that you are legally allowed to download, process, and redistribute.

## Accuracy Notes

The application answers questions using the selected documents. It does not independently verify whether a document is current, complete, or legally authoritative.

Answer quality depends on:

- Document quality
- OCR and text extraction quality
- Chunking
- Retrieval accuracy
- Selected documents
- Ollama model capability
- Available memory and processing power

Always verify important information against the original source.

## Disclaimer

This application provides educational information from user-supplied documents. It is not a substitute for professional legal, medical, financial, immigration, security, or other regulated advice.