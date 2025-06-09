#!/usr/bin/env python3
# app.py

import os
import faiss
import json
import numpy as np
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

# Carrega configuração e modelos
load_dotenv()  # garante OPENAI_API_KEY
client = OpenAI()

# 1) Carrega índice FAISS e chunks
index = faiss.read_index("faiss.index")
with open("chunks.json", encoding="utf-8") as f:
    docs = json.load(f)
chunk_map = { d["id"]: d["text"] for d in docs }
id_list   = [ d["id"]       for d in docs ]

# 2) Função de recuperação
def retrieve(query: str, k: int = 5):
    resp  = client.embeddings.create(model="text-embedding-ada-002", input=[query])
    q_emb = np.array(resp.data[0].embedding, dtype="float32")[None, :]
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    return [ (id_list[i], float(D[0][j])) for j, i in enumerate(I[0]) ]

# 3) Função de resposta
def answer(query: str):
    hits = retrieve(query)
    prompt_chunks = []
    for cid, score in hits:
        text = chunk_map[cid]
        prompt_chunks.append(f"=== (score: {score:.3f}) [{cid}]\n{text}\n")

    context = "\n".join(prompt_chunks)
    prompt  = (
        "Você é um assistente jurídico. Com base nos trechos abaixo, responda em português:\n\n"
        f"{context}\nPergunta: {query}\nResposta:"
    )

    chat = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user", "content":prompt}],
        temperature=0
    )
    return chat.choices[0].message.content

# 4) Interface Gradio
iface = gr.Interface(
    fn=answer,
    inputs=gr.Textbox(lines=2, placeholder="Digite sua pergunta aqui...", label="Pergunta"),
    outputs=gr.Textbox(lines=10, label="Resposta"),
    title="MemoryRAG Test",
    description="Faça perguntas sobre seus documentos jurídicos em Português-BR.",
    allow_flagging="never"
)

if __name__ == "__main__":
    # Para acessar externamente use: share=True
    iface.launch(server_name="0.0.0.0", server_port=7860,)
