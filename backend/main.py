from dotenv import load_dotenv

load_dotenv()  # must run BEFORE importing rag, so the .env values are available

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

import rag

ALLOWED = {".pdf", ".docx", ".txt", ".md"}
MAX_MB = 20

app = FastAPI(title="Intelligent University Knowledge Assistant")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)
    filename: str | None = None      # optional: search only inside one document


@app.get("/api/documents")
def documents():
    return rag.list_documents()


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    results = []
    for f in files:
        name = os.path.basename(f.filename or "")
        if Path(name).suffix.lower() not in ALLOWED:
            results.append({"filename": name, "error": "Only PDF, DOCX, TXT and MD files are supported."})
            continue
        data = await f.read()
        if len(data) > MAX_MB * 1024 * 1024:
            results.append({"filename": name, "error": f"File is larger than {MAX_MB} MB."})
            continue
        try:
            results.append(await run_in_threadpool(rag.ingest, name, data))
        except ValueError as e:
            results.append({"filename": name, "error": str(e)})
    return results


@app.delete("/api/documents/{filename}")
def delete(filename: str):
    removed = rag.delete_document(filename)
    if not removed:
        raise HTTPException(404, "Document not found.")
    return {"filename": filename, "chunks_removed": removed}


@app.post("/api/ask")
async def ask(req: AskRequest):
    return await run_in_threadpool(rag.answer, req.question.strip(), req.top_k, req.filename)


# Serves the frontend at http://localhost:8000  (keep this LAST, after the API routes)
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
