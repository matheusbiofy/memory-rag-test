import json
import os
from typing import List, Dict

import numpy as np
from openai import OpenAI

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # sentence-transformers not installed
    SentenceTransformer = None

# Paths for caches
EMBED_CACHE_PATH = os.getenv("EMBED_CACHE", "embed_cache.json")
RESP_CACHE_PATH = os.getenv("RESP_CACHE", "completion_cache.json")

# Load caches
if os.path.exists(EMBED_CACHE_PATH):
    with open(EMBED_CACHE_PATH, "r", encoding="utf-8") as f:
        _EMBED_CACHE: Dict[str, List[float]] = json.load(f)
else:
    _EMBED_CACHE = {}

if os.path.exists(RESP_CACHE_PATH):
    with open(RESP_CACHE_PATH, "r", encoding="utf-8") as f:
        _RESP_CACHE: Dict[str, str] = json.load(f)
else:
    _RESP_CACHE = {}

client = OpenAI()

local_model_name = os.getenv("LOCAL_EMBED_MODEL")
_local_model = SentenceTransformer(local_model_name) if local_model_name and SentenceTransformer else None


def save_caches() -> None:
    with open(EMBED_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(_EMBED_CACHE, f)
    with open(RESP_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(_RESP_CACHE, f)


def get_embedding(text: str) -> np.ndarray:
    if text in _EMBED_CACHE:
        return np.array(_EMBED_CACHE[text], dtype="float32")
    if _local_model:
        emb = _local_model.encode([text])[0].tolist()
    else:
        resp = client.embeddings.create(model="text-embedding-ada-002", input=[text])
        emb = resp.data[0].embedding
    _EMBED_CACHE[text] = emb
    save_caches()
    return np.array(emb, dtype="float32")


def cached_completion(prompt: str) -> str:
    if prompt in _RESP_CACHE:
        return _RESP_CACHE[prompt]
    chat = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = chat.choices[0].message.content
    _RESP_CACHE[prompt] = answer
    save_caches()
    return answer
