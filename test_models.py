import requests
from app.core.config import settings

url = settings.EXTERNAL_LLM_API_URL.replace("/generate", "/models")
headers = {"api-key": settings.EXTERNAL_LLM_API_KEY}
print(f"Querying {url}")
res = requests.get(url, headers=headers)
print(res.status_code)
print(res.text)
