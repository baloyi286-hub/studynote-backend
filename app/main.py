import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from app.services import ocr, questions

app = FastAPI(title="StudyNote AI API", version="0.1.0")
origins = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "http://localhost:8080").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])

class QuestionsRequest(BaseModel):
    notes: str = Field(min_length=20, max_length=30000)
    count: int = Field(default=10, ge=1, le=30)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/v1/ocr")
async def extract_text(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "Upload a JPEG, PNG, or WebP image")
    content = await file.read(12 * 1024 * 1024 + 1)
    if len(content) > 12 * 1024 * 1024:
        raise HTTPException(413, "Image exceeds 12 MB")
    try:
        return {"text": await ocr.extract(content)}
    except Exception as exc:
        raise HTTPException(502, f"OCR inference failed: {type(exc).__name__}") from exc

@app.post("/api/v1/questions")
async def generate_questions(request: QuestionsRequest):
    try:
        return await questions.generate(request.notes, request.count)
    except Exception as exc:
        raise HTTPException(502, f"Question inference failed: {type(exc).__name__}") from exc
