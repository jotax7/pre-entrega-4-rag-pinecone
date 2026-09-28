"""
ingest.py
---------
Pipeline de ingesta: carga los documentos de data/, los chunkea (ver
document_loader.py), genera embeddings con Google AI Studio y los sube a
Pinecone. El texto original queda guardado en la metadata de cada vector
(langchain_pinecone lo hace automáticamente bajo la key "text"), así que
después no hace falta pegarle a ninguna otra base de datos para mostrar
el contenido recuperado.

Uso:
    python ingest.py            # ingesta normal (agrega/actualiza)
    python ingest.py --force    # borra todo el namespace y reingesta desde cero
"""

import argparse
import time

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

import config
from document_loader import load_and_chunk
from init_pinecone import ensure_index_exists

# El free tier de gemini-embedding-2 tira 429 tanto en batch_embed_contents
# como en RPM. Vamos de a 1 chunk con pausa fija: ~10 RPM, muy por debajo
# de cualquier cuota razonable. Sin reintentos: si falla, falla.
EMBED_BATCH = 1
BATCH_PAUSE_SECONDS = 7


def _get_embeddings():
    return config.get_embeddings()


def _clear_namespace() -> None:
    """Borra todos los vectores del namespace antes de reingestar.
    Esto es lo que hace que --force sea idempotente: podés correrlo
    las veces que quieras y nunca vas a terminar con duplicados."""
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index(config.INDEX_NAME)
    print(f"🗑️  Borrando namespace '{config.PINECONE_NAMESPACE}'...")
    try:
        index.delete(delete_all=True, namespace=config.PINECONE_NAMESPACE)
    except Exception as e:
        # Pinecone tira error si el namespace todavía no tiene datos
        # (nada que borrar la primera vez). Lo ignoramos.
        print(f"   (nada para borrar o namespace nuevo: {e})")


def run_ingestion(force: bool = False) -> None:
    config.validate_env()
    ensure_index_exists()

    if force:
        _clear_namespace()

    print("📄 Cargando y chunkeando documentos...")
    chunks = load_and_chunk()
    print(f"   {len(chunks)} chunks generados desde {config.DATA_DIR}/")

    print("🧠 Generando embeddings y subiendo a Pinecone "
          f"(batches de {EMBED_BATCH}, pausa {BATCH_PAUSE_SECONDS}s)...")
    embeddings = _get_embeddings()
    vectorstore = PineconeVectorStore(
        index_name=config.INDEX_NAME,
        embedding=embeddings,
        namespace=config.PINECONE_NAMESPACE,
    )

    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start:start + EMBED_BATCH]
        end = start + len(batch)
        print(f"   upsert {end}/{len(chunks)}")
        vectorstore.add_documents(batch)
        if end < len(chunks):
            time.sleep(BATCH_PAUSE_SECONDS)

    print(f"✅ Ingesta completa: {len(chunks)} chunks en el namespace "
          f"'{config.PINECONE_NAMESPACE}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingesta de documentos a Pinecone")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Borra el namespace antes de reingestar (evita duplicados)",
    )
    args = parser.parse_args()
    run_ingestion(force=args.force)
