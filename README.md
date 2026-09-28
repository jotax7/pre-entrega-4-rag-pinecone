# Pre-entrega 4 — Sistema RAG escalable en la nube con Pinecone

Módulo de recuperación escalable: pipeline de ingesta a Pinecone,
recuperador híbrido (BM25 + vectorial) y script de evaluación con
Recall@5 / Precision@5.

## Decisiones de diseño

| Decisión | Elegido | Por qué |
|---|---|---|
| Embeddings | Google AI Studio `gemini-embedding-2`, truncado a 768 dim (MRL) | Gratis, sin depender de OpenAI/Anthropic. `text-embedding-004` quedó deprecado (404 en v1beta) y `gemini-embedding-001` está fuera del free tier (429 con cuota 0), así que se usa `gemini-embedding-2` forzando `output_dimensionality=768` para no tener que recrear el índice de Pinecone. |
| Vector DB | Pinecone Serverless, `cosine`, dim=768 | Pedido por la consigna. |
| Chunking | `RecursiveCharacterTextSplitter`, tamaño medido en **tokens** (tiktoken `cl100k_base`), objetivo 650 tokens / 100 overlap | Evita el error típico de medir en caracteres. Sobre el corpus actual: promedio 556 tokens, máximo 630; algunos chunks finales de documento quedan bastante más chicos (mínimo 112) porque el splitter corta por separadores y no rellena el último fragmento. |
| Retriever léxico | `BM25Retriever` sobre el mismo corpus chunkeado (en memoria, sin infra extra) | Captura términos técnicos exactos que el embedding puede diluir. |
| Combinación | `EnsembleRetriever`, pesos `[0.4 BM25, 0.6 vectorial]` | Da algo más de peso a la semántica, sin perder precisión léxica. |
| Namespace | `docs-python-libs` | Evita ruido si más adelante se indexan otros tipos de datos en el mismo índice. |

No se usa LLM de generación en esta entrega: la consigna pide solo
recuperación + evaluación, así que no hace falta Groq acá. Si en una
próxima etapa se agrega generación de respuesta final, ahí entra Groq como
motor de inferencia.

## Estructura del repo

```
.
├── config.py              # variables de entorno y constantes
├── document_loader.py     # carga + chunking determinístico (compartido)
├── init_pinecone.py       # crea el índice si no existe
├── ingest.py              # pipeline de ingesta (--force para reingestar)
├── rag_system.py          # clase RAGSystem (EnsembleRetriever)
├── evaluate.py            # Recall@5 / Precision@5 sobre golden_set.json
├── golden_set.json        # benchmark de 5 preguntas
└── data/                  # 15 .md de la documentación de FastAPI (tutorial + advanced)
```

## Setup y replicación del índice

1. **Instalar dependencias**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configurar credenciales**
   ```bash
   cp .env.example .env
   ```
   Completar en `.env`:
   - `GOOGLE_API_KEY`: API key de [Google AI Studio](https://aistudio.google.com/) (gratis).
   - `PINECONE_API_KEY`: API key de [Pinecone](https://www.pinecone.io/) (free tier).
   - `INDEX_NAME`, `PINECONE_CLOUD`, `PINECONE_REGION`: se puede dejar el default (`aws` / `us-east-1`, requerido por el free tier de Pinecone).

3. **Crear el índice** (se puede saltear este paso: `ingest.py` lo llama automáticamente)
   ```bash
   python init_pinecone.py
   ```
   Esto crea un índice Serverless llamado `pre-entrega-4-rag`, dimensión
   768, métrica coseno. Si ya existe, no hace nada.

4. **Ingestar los documentos**
   ```bash
   python ingest.py --force
   ```
   `--force` borra el namespace antes de subir, para que correrlo varias
   veces nunca genere duplicados.

5. **Probar el recuperador**
   ```bash
   python rag_system.py
   ```

6. **Correr la evaluación**
   ```bash
   python evaluate.py
   ```

## Corpus

- **Fuente:** 15 archivos `.md` de la documentación oficial de
  [FastAPI](https://github.com/fastapi/fastapi) (mezcla de `tutorial/` y
  `advanced/`, incluidos OAuth2 con JWT, WebSockets, SQL, BackgroundTasks,
  APIRouter, dependencias con yield, etc.).
- **Licencia del corpus:** MIT (FastAPI, © Sebastián Ramírez).
- **Descarga:** los archivos se levantaron directamente de
  `raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/` y se
  guardaron con el prefijo `fastapi_` en `data/`.
- **Chunking:** 66 chunks totales tras aplicar el splitter
  (`RecursiveCharacterTextSplitter` medido en tokens `cl100k_base`,
  chunk_size=650, overlap=100). Token count por chunk: **min=112,
  avg=556.1, max=630** (el min bajo corresponde a los chunks finales de
  archivos cortos como `static-files.md`).
- **Metadatos por chunk** (se guardan en cada vector de Pinecone y también
  en los `Document` que consume BM25):
  - `source`: nombre del `.md` original (ej. `fastapi_advanced_websockets.md`).
    Es lo que compara `evaluate.py` contra `documento_id_esperado`.
  - `chunk_index`: posición del chunk dentro del archivo (empieza en 0).
  - `category`: derivada del prefijo del archivo — `"tutorial"` para
    `fastapi_tutorial_*`, `"advanced"` para `fastapi_advanced_*`, y
    `"other"` como fallback. Sobre el corpus actual quedan 57 chunks
    `tutorial` y 9 `advanced`. Útil para filtrar por sección en el futuro
    sin re-ingestar.
  - `token_count`: largo del chunk en tokens `cl100k_base`.

## Ingesta con rate limit del free tier

El free tier de `gemini-embedding-2` rechaza `batch_embed_contents` con
muchos ítems y también aplica un límite por minuto. Por eso `ingest.py`
sube los chunks de a **1 por vez** con una pausa fija de **7 segundos**
entre uno y el siguiente (~10 RPM, muy por debajo de cualquier cuota
razonable). Sin reintentos: si un chunk falla, el pipeline corta.
Los 66 chunks tardan ~8 minutos en subirse.

## Resultado de la evaluación

Ejecutando `python evaluate.py` sobre el índice recién ingestado:

```
Recall@5 promedio:    100.00%
Precision@5 promedio:  80.00%
```

Detalle por pregunta:

| # | Pregunta (fuente esperada) | Recall@5 | Precision@5 |
|---|---|---|---|
| 1 | Librería para hashear passwords (OAuth2+JWT) → `fastapi_tutorial_security_oauth2-jwt.md` | 1.00 | 1.00 |
| 2 | Dependencia con setup/cleanup vía `yield` → `fastapi_tutorial_dependencies_dependencies-with-yield.md` | 1.00 | 1.00 |
| 3 | `APIRouter` para separar rutas en varios archivos → `fastapi_tutorial_bigger-applications.md` | 1.00 | 1.00 |
| 4 | Endpoint WebSocket + aceptar conexiones → `fastapi_advanced_websockets.md` | 1.00 | 0.60 |
| 5 | `BackgroundTasks` para tareas post-respuesta → `fastapi_tutorial_background-tasks.md` | 1.00 | 0.40 |

**Sobre Precision@5:** cuenta cuántos de los 5 chunks devueltos vienen
del `source` esperado en el golden set (`documento_id_esperado`, marcado
como la ÚNICA fuente relevante). Está topeada por la cantidad de chunks
que ese archivo aporta al corpus: `websockets.md` tiene 3 chunks (techo
0.60) y `background-tasks.md` tiene 2 (techo 0.40). Esos dos casos ya
están en el techo. En un benchmark serio convendría marcar múltiples
docs relevantes por pregunta.

## Notas

- El corpus de `data/` es intercambiable: `document_loader.py` levanta
  todos los `.md` que encuentre en esa carpeta, así que se puede
  reemplazar por otra documentación técnica sin tocar código.
- `document_loader.py` es compartido entre `ingest.py` y `rag_system.py`
  para garantizar que el corpus de BM25 (en memoria) sea un espejo exacto
  de lo que está indexado en Pinecone.
