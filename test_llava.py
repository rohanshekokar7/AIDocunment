import httpx
import base64
import json
from app.core.config import settings

def test():
    # A tiny 1x1 white pixel in base64
    img_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
    
    headers = {"api-key": settings.EXTERNAL_LLM_API_KEY}
    payload = {
        "model": "llava:7b",
        "query": "What is in this image?",
        "images": [img_b64]
    }
    
    try:
        resp = httpx.post(settings.EXTERNAL_LLM_API_URL, headers=headers, json=payload, timeout=20)
        print("STATUS:", resp.status_code)
        print("TEXT:", resp.text[:200])
    except Exception as e:
        print("ERROR:", e)

test()
