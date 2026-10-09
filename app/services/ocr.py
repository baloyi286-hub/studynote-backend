"""PaddleOCR-VL-1.5 inference adapter.

Point PADDLE_OCR_URL at a GPU-hosted OpenAI-compatible vision endpoint.
The serving implementation must expose PaddleOCR-VL-1.5 and support image inputs.
"""
import base64
import os
import httpx

async def extract(image: bytes) -> str:
    url = os.environ["PADDLE_OCR_URL"].rstrip("/")
    model = os.getenv("PADDLE_OCR_MODEL", "PaddleOCR-VL-1.5")
    mime = "image/png" if image.startswith(b"\\x89PNG") else "image/jpeg"
    encoded = base64.b64encode(image).decode("ascii")
    headers = {}
    if os.getenv("PADDLE_OCR_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['PADDLE_OCR_API_KEY']}"
    payload = {"model": model, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Transcribe every handwritten word faithfully. Preserve line breaks and technical Java/Spring terms. Do not invent missing words."},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}
    ]}], "temperature": 0}
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(f"{url}/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
