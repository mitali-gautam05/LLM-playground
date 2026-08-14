import httpx
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

response = httpx.get(
    "https://router.huggingface.co/v1/models",
    headers={"Authorization": f"Bearer {token}"}
)
models = response.json()

for m in models.get("data", [])[:30]:
    print(m["id"])