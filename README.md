# MemoryRAG Test

Este projeto demonstra um sistema simples de Retrieval-Augmented Generation (RAG) com suporte a memória conversacional efêmera. Os documentos em `./docs` são indexados com FAISS e as interações são armazenadas em sessões persistentes.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # veja abaixo
```

Dependências principais:
- `openai`
- `faiss-cpu`
- `gradio`
- `python-dotenv`
- `langchain`
- `sentence-transformers` (opcional, para embeddings locais)

Crie um arquivo `.env` com sua `OPENAI_API_KEY`.

## Preparando os documentos

Coloque seus arquivos `.txt` ou `.md` em `docs/` e execute:

```bash
python ingest.py
```

Os chunks serão armazenados em `chunks.json` e o índice em `faiss.index`.

## Executando

```bash
python app.py
```

Use `SESSION_ID=myid python app.py` para continuar uma sessão existente. O histórico é gravado em `sessions/<id>.json`.

## Controles de custo

O projeto possui cache de embeddings e respostas em `embed_cache.json` e `completion_cache.json`. Defina `LOCAL_EMBED_MODEL` para usar um modelo do `sentence-transformers` e evitar chamadas à API para embeddings.

