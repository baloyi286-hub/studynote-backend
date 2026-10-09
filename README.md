# StudyNote AI backend

FastAPI orchestration service for PaddleOCR-VL-1.5 and Qwen3-4B GPU inference endpoints.

## Local run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Set environment variables from `.env.example` in your shell or deployment platform. The service **does not run GPU models locally**; configure compatible hosted inference endpoints before using OCR/questions. Endpoints must support the indicated OpenAI-compatible API format. Do not commit API keys.

- `GET /health`
- `POST /api/v1/ocr` multipart image field `file`
- `POST /api/v1/questions` JSON `{ "notes": "...", "count": 10 }`

Production requires authentication, rate limits, persistent storage, and verified model-serving compatibility.
