"""Qwen3-4B question generation using an OpenAI-compatible GPU endpoint."""
import json
import os
import httpx

async def generate(notes: str, count: int) -> dict:
    url = os.environ["QWEN_API_URL"].rstrip("/")
    headers = {}
    if os.getenv("QWEN_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['QWEN_API_KEY']}"
    prompt = (
        f"Create exactly {count} study questions based ONLY on the following notes. "
        "Return a JSON object with a 'questions' array. Each item must have "
        "'type' (multiple_choice or short_answer), 'question', 'options' (four strings "
        "for multiple_choice, empty array otherwise), 'correctAnswer', and 'explanation'. "
        "Do not fabricate facts. Return JSON only.\n\nNOTES:\n" + notes
    )
    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(f"{url}/chat/completions", headers=headers, json={
            "model": os.getenv("QWEN_MODEL", "Qwen/Qwen3-4B"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        })
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)
