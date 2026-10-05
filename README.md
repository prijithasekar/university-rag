# Intelligent University Knowledge Assistant (RAG Mini Project)

Upload college documents (PDF / DOCX / TXT), ask questions, get answers with source references.

## Setup (line by line)

1. **Python version check pannunga** (3.10 to 3.12 best). Terminal la type pannunga:
   ```
   python --version
   ```
2. **Backend folder kulla pongga**:
   ```
   cd university-rag/backend
   ```
3. **Virtual environment create pannunga** (project packages thani ah irukkum):
   ```
   python -m venv venv
   ```
4. **Virtual environment activate pannunga**:
   - Windows: `venv\Scripts\activate`
   - Mac / Linux: `source venv/bin/activate`
5. **Packages install pannunga** (first time konjam neram aagum):
   ```
   pip install -r requirements.txt
   ```
6. **.env file create pannunga**: `.env.example` ah copy panni `.env` nu rename pannunga.
   - API key irundha `ANTHROPIC_API_KEY=` ku pinnaadi paste pannunga.
   - Key illana kooda app work aagum. Written answer ku badhila best matching lines kaattum.
7. **Server start pannunga**:
   ```
   uvicorn main:app --reload
   ```
   First time embedding model (~80 MB) download aagum. Wait pannunga.
8. **Browser la open pannunga**: http://localhost:8000
9. `sample_docs` folder la irukkura 3 files ah upload pannunga, apram try pannunga:
   - "What is the minimum attendance to write exams?" (2024 and 2025 circular rendum different answer solluthu, so conflict warning varum)
   - "When do hostel gates close?"
   - "Who is the principal?" (documents la illa, so "not found" nu solla vendum)

## How it covers each requirement

| Requirement | Where in code |
|---|---|
| Processing institutional documents | `rag.py` > `extract_pages`, `chunk_text` |
| Creating searchable knowledge repositories | `rag.py` > ChromaDB collection, `ingest` |
| Retrieving from multiple sources | `rag.py` > `retrieve` (searches all files, max 3 chunks per file) |
| Answering using retrieved context | `rag.py` > `answer`, `SYSTEM_PROMPT` |
| Document uploads and updates | `main.py` > `/api/upload` (same file name = replace, version goes up), delete endpoint |
| Explainable responses with source references | `[1] [2]` markers + source cards (file, page, date, similarity) |

## Extra features (for marks)

- **Conflict detection**: when two documents disagree, the answer starts with a warning and names the newer file.
- **"Not found" guard**: if the best match is below `MIN_SIMILARITY`, it says it could not find the answer instead of guessing.
- **Confidence badge**: high / medium / low, from the top similarity score.
- **Search scope**: pick one document from the dropdown to search only inside it.

## Pipeline (viva la idhu explain pannunga)

```
Upload -> extract text per page -> split into 180-word chunks (40 overlap)
      -> embed (MiniLM) -> store in ChromaDB with file, page, upload date

Question -> embed -> top similar chunks (cosine) from all files
         -> chunks + question to LLM ("use only these sources, cite [n]")
         -> answer + source cards
```

## Folder structure

```
university-rag/
  backend/
    main.py            FastAPI routes
    rag.py             full RAG pipeline
    requirements.txt
    .env.example
    static/index.html  frontend (chat, library, source cards)
  sample_docs/         3 demo files
  README.md
```

## Limits to mention honestly

- Scanned PDFs (images) need OCR first, otherwise no text is found.
- Chunk size and `MIN_SIMILARITY` may need tuning for your own documents.
- The conflict check depends on the LLM, so it only works when an API key is set.
