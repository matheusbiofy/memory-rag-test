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

### Como o `id` preserva os chunks para o retrieval

Durante a ingestão cada pedaço de texto recebe um identificador exclusivo
`<arquivo>__<n>`. Isso é definido no `ingest.py`:

```python
for i, chunk in enumerate(chunks):
    docs.append({
        "id": f"{fname}__{i}",
        "text": chunk,
        "metadata": {"source": fname}
    })
```

Esse `id` é gravado em `chunks.json` junto com o texto. Na consulta,
`query.py` carrega o índice FAISS e cria um mapa `id → texto` para recuperar o
conteúdo original:

```python
index = faiss.read_index("faiss.index")
with open("chunks.json", encoding="utf-8") as f:
    docs = json.load(f)
chunk_map = { d["id"]: d["text"] for d in docs }
id_list   = [ d["id"] for d in docs ]
```

Ao pesquisar, o FAISS retorna apenas as posições dos vetores mais próximos.
Usamos `id_list` para obter o `id` correspondente e `chunk_map` para localizar o
texto. Assim, o `id` funciona como uma “ponte de memória” entre o vetor
indexado e o trecho original, possibilitando a exibição correta dos documentos
encontrados.

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

## Usando modelos locais com Ollama

Caso tenha um modelo rodando via [Ollama](https://ollama.com) (por exemplo `llama3:8b`), defina a variável de ambiente `OLLAMA_MODEL` antes de executar o chat. Assim as respostas serão geradas pelo servidor local em vez da API da OpenAI.

```bash
export OLLAMA_MODEL="llama3:8b"
python query.py
```

Você pode configurar o endpoint com `OLLAMA_URL` caso não seja `http://localhost:11434`.

