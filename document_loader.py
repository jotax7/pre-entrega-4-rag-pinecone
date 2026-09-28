"""
document_loader.py
-------------------
Carga los archivos .md de data/ y los divide en chunks medidos en TOKENS
(no caracteres). Esta lógica la usan tanto ingest.py (para subir a Pinecone)
como rag_system.py (para armar el corpus local de BM25).

Es clave que ambos usen EXACTAMENTE esta misma función: si el chunking de
Pinecone y el de BM25 quedan distintos, el EnsembleRetriever compara peras
con manzanas y el ranking combinado deja de tener sentido.
"""

import os
import glob
from typing import List

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config

# Tokenizer usado solo para CONTAR tokens al momento de cortar el texto
# (no para tokenizar de verdad para el modelo, solo como vara de medir).
# Se carga de forma perezosa (lazy) la primera vez que hace falta, en vez de
# al importar el módulo: así, importar document_loader.py no dispara una
# descarga de red, y solo se paga ese costo cuando realmente se va a chunkear.
_encoding = None


def _get_encoding():
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def _token_length(text: str) -> int:
    """Función de longitud en tokens que le pasamos al splitter."""
    return len(_get_encoding().encode(text))


def load_and_chunk(data_dir: str = config.DATA_DIR) -> List[Document]:
    """
    Lee todos los .md de data_dir, los separa en chunks de ~650 tokens
    (con 100 de overlap) y devuelve una lista de Document de LangChain,
    cada uno con metadata: source, chunk_index y category.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE_TOKENS,
        chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
        length_function=_token_length,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: List[Document] = []
    paths = sorted(glob.glob(os.path.join(data_dir, "*.md")))

    if not paths:
        raise FileNotFoundError(
            f"No encontré archivos .md en '{data_dir}/'. Poné ahí la "
            f"documentación técnica que quieras indexar."
        )

    for path in paths:
        source_name = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        chunks = splitter.split_text(raw_text)

        for i, chunk_text in enumerate(chunks):
            all_chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "source": source_name,
                        "chunk_index": i,
                        "category": "python-library-docs",
                        "token_count": _token_length(chunk_text),
                    },
                )
            )

    return all_chunks


if __name__ == "__main__":
    # Prueba rápida manual: python document_loader.py
    docs = load_and_chunk()
    print(f"Se generaron {len(docs)} chunks a partir de {config.DATA_DIR}/")
    for d in docs[:3]:
        print(f"- {d.metadata['source']} #{d.metadata['chunk_index']} "
              f"({d.metadata['token_count']} tokens)")
