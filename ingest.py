#!/usr/bin/env python3
# ingest.py
# ------------
# 1. Load all .txt files from ./docs
# 2. Chunk them with overlap
# 3. Persist chunks.json (id, text, metadata)
# 4. Embed in batches via OpenAI
# 5. Build and save FAISS index
# 6. Persist metadatas.json (id → metadata)

import os
import json

import numpy as np
import faiss

from dotenv import load_dotenv
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

def main():
    # Load your OpenAI API key from .env
    load_dotenv()
    client = OpenAI()

    # 1. Set up text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    # 2. Read & chunk all docs
    docs = []
    for fname in os.listdir("docs"):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join("docs", fname)
        text = open(path, encoding="utf-8").read()
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append({
                "id": f"{fname}__{i}",
                "text": chunk,
                "metadata": {"source": fname}
            })

    # 3. Persist the exact chunks for later lookup
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"✅ Wrote chunks.json with {len(docs)} entries")

    # 4. Embed in batches
    texts = [d["text"] for d in docs]
    BATCH = 50
    embeddings = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        resp  = client.embeddings.create(
            model="text-embedding-ada-002",
            input=batch
        )
        # resp.data is a list of objects with .embedding
        embeddings.extend([item.embedding for item in resp.data])

    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    # 5. Build FAISS index
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"✅ Indexed {index.ntotal} chunks")

    # 6. Persist FAISS index and metadata map
    faiss.write_index(index, "faiss.index")
    with open("metadatas.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"id": d["id"], "meta": d["metadata"]} for d in docs],
            f, ensure_ascii=False, indent=2
        )
    print("✅ Wrote metadatas.json")

if __name__ == "__main__":
    main()
