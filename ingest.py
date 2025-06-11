#!/usr/bin/env python3
# ingest.py
# ------------
# 1. Load all text files from ./docs
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
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils import get_embedding, flush_cache

def main():
    # Load your OpenAI API key from .env
    load_dotenv()

    # 1. Set up text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    # 2. Read & chunk all docs
    docs = []
    for fname in sorted(os.listdir("docs")):
        if not (fname.endswith(".txt") or fname.endswith(".md")):
            continue
        path = os.path.join("docs", fname)
        with open(path, encoding="utf-8") as f:
            text = f.read()
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

    if not docs:
        print("⚠️ Nenhum documento encontrado em 'docs/'.")
        return

    # 4. Embed texts with caching to save OpenAI costs
    embeddings = [get_embedding(d["text"]).tolist() for d in docs]

    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    # 5. Build FAISS index
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(x=embeddings)#type: ignore
    print(f"✅ Indexed {index.ntotal} chunks")

    # 6. Persist FAISS index and metadata map, here is what happens the indexing of memory index
    faiss.write_index(index, "faiss.index")
    with open("metadatas.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"id": d["id"], "meta": d["metadata"]} for d in docs],
            f, ensure_ascii=False, indent=2
        )
    print("✅ Wrote metadatas.json")

    # Persist caches only once after all embeddings are processed
    flush_cache()

if __name__ == "__main__":
    main()
