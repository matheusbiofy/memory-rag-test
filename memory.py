import json
import os
import uuid
from typing import List

import faiss
import numpy as np
from openai import OpenAI

from utils import get_embedding


class EphemeralMemory:
    """Simple ephemeral memory with summarization and persistence."""

    def __init__(self, session_id: str | None = None, max_history: int = 10):
        self.client = OpenAI()
        self.max_history = max_history
        self.session_id = session_id or uuid.uuid4().hex
        os.makedirs("sessions", exist_ok=True)
        self.session_path = os.path.join("sessions", f"{self.session_id}.json")
        if os.path.exists(self.session_path):
            with open(self.session_path, encoding="utf-8") as f:
                self.history = json.load(f)
        else:
            self.history = []
            self._persist()

        # Pre-compute embeddings for loaded history
        self.embeddings: list[np.ndarray] = []
        for msg in self.history:
            emb = get_embedding(msg["content"])
            faiss.normalize_L2(emb[None, :])
            self.embeddings.append(emb)

    def _persist(self) -> None:
        with open(self.session_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def add(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        emb = get_embedding(content)
        faiss.normalize_L2(emb[None, :])
        self.embeddings.append(emb)
        if len(self.history) > self.max_history:
            self._summarize()
        self._persist()

    def _summarize(self) -> None:
        """Summarize the oldest messages to keep memory short."""
        prefix = self.history[:4]
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in prefix)
        prompt = (
            "Resuma a seguinte conversa em português, mantendo as informações essenciais:\n"
            f"{convo}\nResumo:"
        )
        chat = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        summary = chat.choices[0].message.content.strip()  # type: ignore
        self.history = [{"role": "system", "content": summary}] + self.history[4:]
        emb = get_embedding(summary)
        faiss.normalize_L2(emb[None, :])
        self.embeddings = [emb] + self.embeddings[4:]

    def retrieve(self, query: str, top_k: int = 2) -> List[str]:
        if not self.history:
            return []
        q_emb = get_embedding(query)
        faiss.normalize_L2(q_emb[None, :])
        mem_embs = np.vstack(self.embeddings)
        scores = mem_embs @ q_emb
        idx = np.argsort(scores)[::-1][:top_k]
        return [self.history[i]["content"] for i in idx]
