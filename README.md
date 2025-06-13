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
Os caches de embeddings agora são salvos apenas ao final da ingestão para reduzir escritas em disco.

## Executando

```bash
python query.py
```

Use `SESSION_ID=myid python query.py` para continuar uma sessão existente. O histórico é gravado em `sessions/<id>.json`.

Ao executar, será aberta uma interface de **chat** no navegador. Todas as mensagens trocadas ficam visíveis e são usadas para compor novas respostas.

## Controles de custo

O projeto possui cache de embeddings e respostas em `embed_cache.json` e `completion_cache.json`. Defina `LOCAL_EMBED_MODEL` para usar um modelo do `sentence-transformers` e evitar chamadas à API para embeddings.
Durante a ingestão, os embeddings são persistidos em lote somente ao final do processo.

Se esses arquivos ficarem corrompidos (por exemplo, erros de JSON), apague-os para que sejam recriados automaticamente.

