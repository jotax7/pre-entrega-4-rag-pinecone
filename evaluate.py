"""
evaluate.py
-----------
Corre el golden_set.json contra el RAGSystem y calcula dos métricas:

- Recall@5: de las preguntas evaluadas, ¿en qué fracción el documento
  fuente esperado aparece ENTRE los 5 resultados recuperados? (0 o 1 por
  pregunta, promediado).
- Precision@5: de los 5 documentos recuperados por pregunta, ¿qué
  proporción pertenece realmente al documento fuente esperado?

Nota sobre la métrica: el golden_set solo marca UN documento fuente como
relevante por pregunta (formato que pide la consigna:
{"pregunta": ..., "documento_id_esperado": ...}). Por eso Precision@5 acá es
estricta: cuenta cuántos de los 5 chunks devueltos vienen de ESE mismo
archivo fuente. Es una simplificación razonable para un golden set chico,
pero en un benchmark más grande convendría marcar varios documentos
relevantes por pregunta.

Uso:
    python evaluate.py
"""

import json

from rag_system import RAGSystem


def load_golden_set(path: str = "golden_set.json") -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate() -> None:
    golden_set = load_golden_set()
    rag = RAGSystem()

    recalls = []
    precisions = []

    print("=" * 70)
    print("EVALUACIÓN DEL SISTEMA RAG — Recall@5 / Precision@5")
    print("=" * 70)

    for item in golden_set:
        pregunta = item["pregunta"]
        fuente_esperada = item["documento_id_esperado"]

        resultados = rag.query(pregunta)
        fuentes_recuperadas = [d.metadata.get("source") for d in resultados]

        hit = fuente_esperada in fuentes_recuperadas
        recall = 1.0 if hit else 0.0

        aciertos = sum(1 for f in fuentes_recuperadas if f == fuente_esperada)
        precision = aciertos / len(resultados) if resultados else 0.0

        recalls.append(recall)
        precisions.append(precision)

        print(f"\nPregunta: {pregunta}")
        print(f"  Fuente esperada: {fuente_esperada}")
        print(f"  Fuentes recuperadas (top-5): {fuentes_recuperadas}")
        print(f"  Recall@5: {recall:.2f}  |  Precision@5: {precision:.2f}")

    recall_promedio = sum(recalls) / len(recalls)
    precision_promedio = sum(precisions) / len(precisions)

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"Preguntas evaluadas: {len(golden_set)}")
    print(f"Recall@5 promedio:    {recall_promedio:.2%}")
    print(f"Precision@5 promedio: {precision_promedio:.2%}")
    print("=" * 70)


if __name__ == "__main__":
    evaluate()
