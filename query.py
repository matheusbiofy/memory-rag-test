import json
import numpy as np
import faiss
from dotenv import load_dotenv

from utils import get_embedding, cached_completion

load_dotenv()

# 1. Load FAISS index + metadata map
index = faiss.read_index("faiss.index")

# 2. Load chunks.json so we have exact text for each id
with open("chunks.json", encoding="utf-8") as f:
    docs = json.load(f)       # list of {id, text, metadata}
chunk_map = { d["id"]: d["text"] for d in docs }
id_list   = [ d["id"]       for d in docs ]

# 3. Retrieval
def retrieve(query: str, k: int = 5):
    q_emb = get_embedding(query)[None, :]
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k) # retorna os IDs dos trechos mais similares
    # return list of (chunk_id, score)
    return [ (id_list[i], float(D[0][j])) for j, i in enumerate(I[0]) ]

# 4. Answer construction
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

    return cached_completion(prompt)

if __name__ == "__main__":
    q = input("Pergunta: ")
    print("\nResposta:\n", answer(q))
