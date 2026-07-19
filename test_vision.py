import httpx
import base64
import json
from app.core.config import settings

def test():
    with open("dummy.jpg", "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    headers = {"api-key": settings.EXTERNAL_LLM_API_KEY}
    payload = {
        "model": "glm-ocr:latest",
        "query": "Is this text Printed, Handwritten, or Mixed? Answer strictly with one of these words.",
        "images": [img_b64]
    }
    
    resp = httpx.post(settings.EXTERNAL_LLM_API_URL, headers=headers, json=payload, timeout=60)
    print(resp.status_code)
    print(resp.text)

test()
