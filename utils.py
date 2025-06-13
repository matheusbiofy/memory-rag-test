import json
import os
import logging
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
    try:
        with open(EMBED_CACHE_PATH, "r", encoding="utf-8") as f:
            _EMBED_CACHE: Dict[str, List[float]] = json.load(f)
    except json.JSONDecodeError:
        logging.warning("Failed to decode %s, starting with empty embed cache", EMBED_CACHE_PATH)
        _EMBED_CACHE = {}
else:
    _EMBED_CACHE = {}

if os.path.exists(RESP_CACHE_PATH):
    try:
        with open(RESP_CACHE_PATH, "r", encoding="utf-8") as f:
            _RESP_CACHE: Dict[str, str] = json.load(f)
    except json.JSONDecodeError:
        logging.warning("Failed to decode %s, starting with empty completion cache", RESP_CACHE_PATH)
        _RESP_CACHE = {}
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


def flush_cache() -> None:
    """Persist caches to disk."""
    save_caches()


def get_embedding(text: str) -> np.ndarray:
    if text in _EMBED_CACHE:
        return np.array(_EMBED_CACHE[text], dtype="float32")
    if _local_model:
        emb = _local_model.encode([text])[0].tolist()
    else:
        resp = client.embeddings.create(model="text-embedding-ada-002", input=[text])
        emb = resp.data[0].embedding
    _EMBED_CACHE[text] = emb
    return np.array(emb, dtype="float32")


def cached_completion(prompt: str) -> str:
    if prompt in _RESP_CACHE:
        return _RESP_CACHE[prompt]
    system_message = "Você é um assistente jurídico chamado LexIA. Você é um especialista em direito brasileiro, capaz de responder perguntas sobre legislação, jurisprudência e doutrina. Sua missão é fornecer respostas precisas e úteis com base nas informações disponíveis. Use o contexto fornecido para responder às perguntas do usuário. Se não souber a resposta, diga que não sabe e sugira consultar os documentos relevantes que você possui. Sempre responda em Português-BR, mantendo um tom profissional e claro." 
    ""

    chat = client.chat.completions.create(
        model="o4-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
    )
    answer = chat.choices[0].message.content
    if answer is not None:
        _RESP_CACHE[prompt] = answer
        save_caches()
    return answer or ""
