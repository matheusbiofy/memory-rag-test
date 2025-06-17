import json
import os
import logging
from typing import List, Dict
import re

import numpy as np
import requests
from openai import OpenAI

# Configuration for optional Ollama usage
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
# Base URL for an optional Ollama server. The official API endpoints live
# directly under the server root, e.g. `http://localhost:11434/api/...`.
# The previous default incorrectly included a `/v1` prefix which caused
# requests to fail with a 404 error when using Ollama locally.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

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
        resp = client.embeddings.create(model="text-embedding-3-large", input=[text])
        emb = resp.data[0].embedding
    _EMBED_CACHE[text] = emb
    return np.array(emb, dtype="float32")


def cached_completion(prompt: str) -> str:
    if prompt in _RESP_CACHE:
        return _RESP_CACHE[prompt]

    system_message = (
        "Você é um assistente jurídico chamado LexIA, criado pela equipe da Biofy Technologies. Você é um especialista em "
        "direito brasileiro, capaz de responder perguntas sobre legislação, "
        "jurisprudência e doutrina. Sua missão é fornecer respostas precisas e "
        "úteis com base nas informações disponíveis. Use o contexto fornecido para "
        "responder às perguntas do usuário. Se não souber a resposta, diga que não "
        "sabe e sugira consultar os documentos relevantes que você possui. Sempre "
        "responda em Português-BR, mantendo um tom profissional e claro. "
        "Cada resposta que você fornecer deve citar de forma amigável o nome do documento onde a informação aparece (ex.: '05/04/2022 - Deferido o pedido').\n\n"
        "### Exemplos de interação\n"
    )
    #few-shot example 1
    system_message += (
        "Usuário: Quais são as partes envolvidas na execução de título "
        "extrajudicial e qual o valor da dívida?\n"
        "Assistente: As partes são a Empresa Gestora de Ativos (EMGEA) e Antonio "
        "Arnaldo Debona - Espólio, além da De Bona Construções Civis Ltda. O valor "
        "da execução evoluiu de R$ 781.700,38 em 2013 para cerca de "
        "R$ 94.915.840,44 em 2021.\n\n"
    )
    #few-shot example 2
    system_message += (
        "Usuário: Quais bens foram objeto de penhora e leilão neste processo?\n"
        "Assistente: Foram penhorados diversos imóveis dos executados, incluindo "
        "garagens, salas comerciais e terrenos. Alguns foram leiloados, outros "
        "tiveram o leilão cancelado por falta de registro da penhora.\n\n"
    )
    #few-shot example 3
    system_message += (
        "Usuário: Como funciona a preferência de créditos em caso de leilão "
        "judicial?\n"
        "Assistente: Créditos trabalhistas têm prioridade máxima. Em seguida vêm "
        "os créditos tributários, que prevalecem sobre quaisquer outros, "
        "inclusive os garantidos por hipoteca.\n"
    )
    #few-shot example 4
    system_message += (
        "Usuário: Teve leilão cancelado neste processo?\n"
        "Assistente: Sim, o leilão de 10/11/2021 foi cancelado por falta de "
        "registro da penhora dos bens. O juiz determinou a intimação dos "
        "executados para regularizar a situação.\n\n"
        "Usuário: Qual é o valor da dívida atualizada?\n"
        "Assistente: O valor atualizado da dívida é de aproximadamente "
    )
    
    if OLLAMA_MODEL:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": f"{system_message}\n{prompt}", "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("response", "")
    else:
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


def humanize_doc_id(doc_id: str) -> str:
    """Return a human friendly name for a given chunk/document id."""
    base = doc_id.split("__")[0]
    base = os.path.basename(base)
    name = os.path.splitext(base)[0]
    m = re.match(r"(\d{2})(\d{2})(\d{4})-\d{6}-(.+)", name)
    if m:
        day, month, year, slug = m.groups()
        slug = slug.replace("-", " ").replace("_", " ")
        slug = slug.capitalize()
        return f"{day}/{month}/{year} - {slug}"
    return name
