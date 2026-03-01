# CCPA Compliance Analyzer

## Solution Overview

This system analyzes business practice descriptions for California Consumer Privacy Act (CCPA) violations using a **RAG (Retrieval-Augmented Generation)** pipeline:

```text
Input Prompt
│
▼
┌───────────────────────────┐
│     Embedding (MiniLM)    │ ← sentence-transformers/all-MiniLM-L6-v2
└─────────────┬─────────────┘
│
▼
┌───────────────────────────┐
│    FAISS Vector Search    │ ← Retrieves top-K relevant CCPA sections
│    + Keyword Boosting     │ ← Hybrid retrieval for better recall
└─────────────┬─────────────┘
│
▼
┌───────────────────────────┐
│     Page Index Lookup     │ ← Maps sections to PDF page numbers
└─────────────┬─────────────┘
│
▼
┌───────────────────────────┐
│    Qwen2.5-7B-Instruct    │ ← 4-bit quantized LLM (< 8B params)
│      + System Prompt      │ ← Structured output with section citations
└─────────────┬─────────────┘
│
▼
┌───────────────────────────┐
│       JSON Parser +       │ ← Robust parsing with multiple fallbacks
│         Validator         │ ← Section normalization & format enforcement
└─────────────┬─────────────┘
│
▼
{"harmful": bool, "articles": [...]}
Here is the well-formatted Markdown `README.md` for your project. I have cleaned up the formatting, properly structured the tables, and added syntax highlighting for the code blocks.

```markdown
# CCPA Compliance Analyzer

## Solution Overview

This system analyzes business practice descriptions for California Consumer Privacy Act (CCPA) violations using a **RAG (Retrieval-Augmented Generation)** pipeline:

```text
Input Prompt
│
▼
┌───────────────────────────┐
│     Embedding (MiniLM)    │ ← sentence-transformers/all-MiniLM-L6-v2
└─────────────┬─────────────┘
│
▼
┌───────────────────────────┐
│    FAISS Vector Search    │ ← Retrieves top-K relevant CCPA sections
│    + Keyword Boosting     │ ← Hybrid retrieval for better recall
└─────────────┬─────────────┘
│
▼
┌───────────────────────────┐
│     Page Index Lookup     │ ← Maps sections to PDF page numbers
└─────────────┬─────────────┘
│
▼
┌───────────────────────────┐
│    Qwen2.5-7B-Instruct    │ ← 4-bit quantized LLM (< 8B params)
│      + System Prompt      │ ← Structured output with section citations
└─────────────┬─────────────┘
│
▼
┌───────────────────────────┐
│       JSON Parser +       │ ← Robust parsing with multiple fallbacks
│         Validator         │ ← Section normalization & format enforcement
└─────────────┬─────────────┘
│
▼
{"harmful": bool, "articles": [...]}
```

### Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Qwen/Qwen2.5-7B-Instruct (4-bit) | Legal reasoning & classification |
| **Embeddings** | all-MiniLM-L6-v2 | Semantic similarity for retrieval |
| **Vector Store** | FAISS (CPU) | Fast nearest-neighbor search |
| **PDF Parser** | PyMuPDF | Page-level text extraction |
| **API Server** | FastAPI + Uvicorn | HTTP interface |
| **Knowledge Base** | Pre-parsed CCPA sections | Reliable fallback + violation criteria |

### Page Index Feature

The system builds a page-level index of the CCPA statute PDF at startup:
- Each page's text is extracted and stored.
- CCPA section boundaries are identified via regex.
- Sections are mapped to their PDF page ranges.
- Retrieved context includes page references for traceability.

---

## Installation & Setup

### Docker Run Command

Pull and run (primary method):
```bash
docker run --gpus all -p 8000:8000 \
    -e HF_TOKEN=<your_token> \
    yourusername/ccpa-compliance:latest
```

*Note: If the model is not gated (e.g., Qwen2.5), `HF_TOKEN` is optional:*
```bash
docker run --gpus all -p 8000:8000 \
    yourusername/ccpa-compliance:latest
```

### Build from Source

```bash
docker build -t ccpa-compliance:latest .
docker run --gpus all -p 8000:8000 ccpa-compliance:latest
```

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | No* | `None` | HuggingFace access token for gated models. |
| `LLM_MODEL` | No | `Qwen/Qwen2.5-7B-Instruct` | LLM model name. |
| `EMBEDDING_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model. |
| `PORT` | No | `8000` | Server port. |
| `TOP_K` | No | `6` | Number of sections to retrieve. |
| `TEMPERATURE` | No | `0.1` | LLM generation temperature. |
| `PDF_PATH` | No | `/app/data/ccpa_statute.pdf` | Path to CCPA PDF. |

*\*Required only if using a gated model like `meta-llama/Meta-Llama-3.1-8B-Instruct`.*

### GPU Requirements

| Configuration | VRAM Required | Notes |
|---|---|---|
| **Qwen2.5-7B 4-bit** | ~6 GB | Recommended |
| **Qwen2.5-7B FP16** | ~16 GB | Higher quality, needs large GPU |
| **Phi-3.5-mini 4-bit** | ~3 GB | Fallback for limited GPU |

**Warning:** CPU-only fallback is NOT recommended — inference will be extremely slow (minutes per request). To attempt CPU-only, run:
 ```bash
docker run -p 8000:8000 -e CUDA_VISIBLE_DEVICES="" ccpa-compliance:latest
 ```

## Local Setup Instructions (Fallback)

If Docker fails, run directly on a Linux VM:

```bash
# 1. System requirements
# Python 3.10+, CUDA 12.1+, 8GB+ GPU VRAM

# 2. Clone and set up
cd ccpa-compliance
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the CCPA PDF
mkdir -p data
cp /path/to/ccpa_statute.pdf data/

# 5. (Optional) Set HuggingFace token
export HF_TOKEN=hf_xxxxx

# 6. Start the server
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Server will download models on first start (~5-10 minutes)
# After that, health check will return 200
```

## API Usage Examples

### Health Check
```bash
curl http://localhost:8000/health
```
**Response:** 
```json
{"status": "healthy"}
```

### Analyze — Violation Detected
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "We sell customer browsing history to ad networks without notifying them."}'
```
**Response:** 
```json
{"harmful": true, "articles": ["Section 1798.100", "Section 1798.120"]}
```

### Analyze — No Violation
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "We provide a clear privacy policy and honor all deletion requests."}'
```
**Response:** 
```json
{"harmful": false, "articles": []}
```

### Analyze — Multiple Violations
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "We collect user health data without notice, sell it to third parties, and charge higher prices to users who opt out."}'
```
**Response:** 
```json
{"harmful": true, "articles": ["Section 1798.100", "Section 1798.120", "Section 1798.121", "Section 1798.125"]}
```

## Build & Deploy Quick Reference

**1. Project structure** (ensure these files are in place):
```text
ccpa-compliance/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── analyzer.py
│   ├── ccpa_knowledge.py
│   ├── pdf_processor.py
│   ├── rag_engine.py
│   └── config.py
├── ccpa_statute.pdf          ← place your PDF here
├── download_model.py
├── requirements.txt
├── Dockerfile
├── startup.sh
├── validate_format.py
└── README.md
```

**2. Build the Docker image:**
```bash
docker build -t yourusername/ccpa-compliance:latest .
```

**3. Run the container:**
```bash
docker run --gpus all -p 8000:8000 yourusername/ccpa-compliance:latest
```

**4. Validate** (in another terminal):
```bash
python3 validate_format.py
```

**5. Push to Docker Hub:**
```bash
docker push yourusername/ccpa-compliance:latest
```
