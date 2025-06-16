#!/usr/bin/env python3
# app.py

import os
import faiss
import json
import numpy as np
import gradio as gr
from dotenv import load_dotenv

from memory import EphemeralMemory
from utils import get_embedding, cached_completion, humanize_doc_id

# Carrega configuração e modelos
load_dotenv()  # garante OPENAI_API_KEY
session_id = os.environ.get("SESSION_ID")
memory = EphemeralMemory(session_id=session_id)

# 1) Carrega índice FAISS e chunks
index = faiss.read_index("faiss.index")
with open("chunks.json", encoding="utf-8") as f:
    docs = json.load(f)
chunk_map = { d["id"]: d["text"] for d in docs }
id_list   = [ d["id"]       for d in docs ]

# 2) Função de recuperação
def retrieve_docs(query: str, k: int = 5):
    q_emb = get_embedding(query)[None, :]
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    return [ (id_list[i], float(D[0][j])) for j, i in enumerate(I[0]) ]

# 3) Função de resposta
def answer(query: str) -> str:
    doc_hits = retrieve_docs(query)
    mem_hits = memory.retrieve(query)
    prompt_chunks = []
    for cid, score in doc_hits:
        text = chunk_map[cid]
        display_name = humanize_doc_id(cid)
        prompt_chunks.append(f"=== (score: {score:.3f}) [{display_name}]\n{text}\n")

    doc_context = "\n".join(prompt_chunks)
    mem_context = "\n".join(mem_hits)
    prompt = f"{doc_context}\nPergunta: {query}\nResposta:"

    answer_text = cached_completion(prompt)
    memory.add("user", query)
    memory.add("assistant", answer_text)
    return answer_text


# 4) Interface de chat com histórico
def chat(query: str, history: list[tuple[str, str]]) -> str:
    """Wrapper para usar com gr.ChatInterface."""
    return answer(query)

iface = gr.ChatInterface(
    fn=chat,
    title="MemoryRAG Test",
    description="Faça perguntas sobre seus documentos jurídicos em Português-BR.",
)

if __name__ == "__main__":
    # Para acessar externamente use: share=True
    iface.launch(server_name="0.0.0.0", server_port=7860,)