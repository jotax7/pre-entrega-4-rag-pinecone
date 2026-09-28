"""
init_pinecone.py
-----------------
Verifica si el índice de Pinecone ya existe y, si no, lo crea en modo
Serverless con la dimensión correcta para los embeddings de Google
(text-embedding-004 -> 768 dims).

Uso:
    python init_pinecone.py
"""

from pinecone import Pinecone, ServerlessSpec

import config


def ensure_index_exists() -> None:
    config.validate_env()
    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    if config.INDEX_NAME in existing_indexes:
        print(f"✅ El índice '{config.INDEX_NAME}' ya existe. No hago nada.")
        return

    print(f"⏳ Creando índice '{config.INDEX_NAME}' "
          f"(dim={config.EMBEDDING_DIMENSION}, cloud={config.PINECONE_CLOUD}, "
          f"region={config.PINECONE_REGION})...")

    pc.create_index(
        name=config.INDEX_NAME,
        dimension=config.EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=config.PINECONE_CLOUD,
            region=config.PINECONE_REGION,
        ),
    )

    print("✅ Índice creado.")


if __name__ == "__main__":
    ensure_index_exists()
