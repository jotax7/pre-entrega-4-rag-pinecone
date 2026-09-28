"""
rag_system.py
-------------
Define RAGSystem, la clase que encapsula el recuperador híbrido.

Combina:
- Búsqueda semántica (vectorial) contra Pinecone -> buena para similitud
  de significado, parafraseos, preguntas conceptuales.
- Búsqueda léxica (BM25) contra un corpus local -> buena para términos
  técnicos exactos, nombres propios, nombres de funciones/parámetros que
  el embedding puede "diluir".

El BM25 se arma re-chunkeando data/ con la MISMA función que usa ingest.py
(document_loader.load_and_chunk), así el corpus léxico es un espejo exacto
de lo que hay en Pinecone y el ranking combinado es consistente.
"""

from typing import List

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_pinecone import PineconeVectorStore

import config
from document_loader import load_and_chunk


class RAGSystem:
    def __init__(self):
        config.validate_env()

        # --- Rama semántica: Pinecone ---
        embeddings = config.get_embeddings()
        vectorstore = PineconeVectorStore(
            index_name=config.INDEX_NAME,
            embedding=embeddings,
            namespace=config.PINECONE_NAMESPACE,
        )
        vector_retriever = vectorstore.as_retriever(
            search_kwargs={"k": config.TOP_K}
        )

        # --- Rama léxica: BM25 (corre en memoria, no pega a ninguna API) ---
        local_chunks = load_and_chunk()
        bm25_retriever = BM25Retriever.from_documents(local_chunks)
        bm25_retriever.k = config.TOP_K

        # --- Combinación ---
        self.retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=config.ENSEMBLE_WEIGHTS,
        )

    def query(self, question: str) -> List[Document]:
        """Devuelve hasta TOP_K documentos combinando ambas búsquedas."""
        results = self.retriever.invoke(question)
        return results[: config.TOP_K]


if __name__ == "__main__":
    # Prueba manual rápida: python rag_system.py
    rag = RAGSystem()
    pregunta = "¿Cómo se define un endpoint de WebSocket en FastAPI?"
    docs = rag.query(pregunta)
    print(f"Pregunta: {pregunta}\n")
    for i, d in enumerate(docs, 1):
        print(f"{i}. [{d.metadata.get('source')}] {d.page_content[:120]}...")
