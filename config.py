"""
config.py
---------
Centraliza toda la configuración del proyecto: lee las variables de entorno
y define las constantes que usan el resto de los scripts. Así evitamos tener
"magic numbers" repetidos en ingest.py, rag_system.py y evaluate.py.
"""

import os
from dotenv import load_dotenv

# Carga el archivo .env a las variables de entorno del proceso
load_dotenv()

# --- Credenciales / infraestructura ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "pre-entrega-4-rag")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "docs-python-libs")

# --- Embeddings ---
# Nota: text-embedding-004 fue deprecado (404 en v1beta). gemini-embedding-001
# existe pero devuelve 429 sin QuotaFailure en el free tier (cuota 0 para
# esta key). gemini-embedding-2 sí está habilitado en el tier gratuito y
# acepta output_dimensionality=768 vía Matryoshka Representation Learning,
# así el índice Pinecone (768 dims) sigue sirviendo sin recrearse.
EMBEDDING_MODEL = "models/gemini-embedding-2"
EMBEDDING_DIMENSION = 768

# --- Chunking ---
# Medimos el tamaño de los chunks en tokens (no en caracteres), usando el
# tokenizer cl100k_base de tiktoken como aproximación razonable (Google no
# expone un tokenizer propio fácil de usar offline). Apuntamos a un objetivo
# de 650 tokens con 100 de overlap; sobre el corpus actual el promedio real
# quedó en 556 tokens y el máximo en 630. Los chunks finales de cada
# documento pueden quedar bastante más cortos (mínimo observado: 112) porque
# RecursiveCharacterTextSplitter corta por separadores y no rellena el
# último fragmento hasta el objetivo.
CHUNK_SIZE_TOKENS = 650
CHUNK_OVERLAP_TOKENS = 100

# --- Carpeta con los documentos fuente ---
DATA_DIR = "data"

# --- Recuperación ---
TOP_K = 5
# Pesos del EnsembleRetriever: [BM25 (léxico), Pinecone (semántico)]
ENSEMBLE_WEIGHTS = [0.4, 0.6]


def get_embeddings():
    """Factory de embeddings. Envuelve GoogleGenerativeAIEmbeddings para
    forzar output_dimensionality=768 en cada llamada (el constructor de
    langchain-google-genai 2.1.x no expone ese parámetro, solo lo aceptan
    los métodos embed_query/embed_documents)."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    class _Google768Embeddings(GoogleGenerativeAIEmbeddings):
        def embed_documents(self, texts, *, task_type=None, titles=None,
                             output_dimensionality=EMBEDDING_DIMENSION,
                             batch_size=100):
            return super().embed_documents(
                texts,
                task_type=task_type,
                titles=titles,
                output_dimensionality=output_dimensionality,
                batch_size=batch_size,
            )

        def embed_query(self, text, *, task_type=None, title=None,
                        output_dimensionality=EMBEDDING_DIMENSION):
            return super().embed_query(
                text,
                task_type=task_type,
                title=title,
                output_dimensionality=output_dimensionality,
            )

    return _Google768Embeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


def validate_env() -> None:
    """Corta rápido si falta alguna variable crítica, en vez de fallar
    a mitad de un pipeline largo con un error críptico de la API."""
    missing = [
        name
        for name, val in [
            ("GOOGLE_API_KEY", GOOGLE_API_KEY),
            ("PINECONE_API_KEY", PINECONE_API_KEY),
        ]
        if not val
    ]
    if missing:
        raise EnvironmentError(
            f"Faltan variables de entorno: {', '.join(missing)}. "
            f"Copiá .env.example a .env y completalas."
        )
