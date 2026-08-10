"""
RAG retrieval quality: precision/recall of retrieved_doc_ids against the
dataset item's relevant_doc_ids.
"""


def rag_precision_recall(retrieved_doc_ids: list, relevant_doc_ids: list) -> dict:
    if not relevant_doc_ids:
        return {
            "precision": None, "recall": None,
            "reason": "No relevant_doc_ids labeled for this item; RAG quality not applicable.",
        }
    retrieved = set(retrieved_doc_ids or [])
    relevant = set(relevant_doc_ids)

    if not retrieved:
        return {"precision": 0.0, "recall": 0.0, "reason": "Agent retrieved no documents."}

    tp = len(retrieved & relevant)
    precision = round(tp / len(retrieved), 3)
    recall = round(tp / len(relevant), 3)
    return {
        "precision": precision,
        "recall": recall,
        "reason": f"{tp} of {len(retrieved)} retrieved docs were relevant; "
                  f"{tp} of {len(relevant)} relevant docs were found.",
    }
