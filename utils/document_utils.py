from rapidfuzz import fuzz
from utils.string_utils import normalize

def find_matching_docs(references, documents, threshold=80):
    matched_docs = []
    normalized_refs = [normalize(ref) for ref in references]

    for doc in documents:
        normalized_doc_text = normalize(doc.content)
        for ref in normalized_refs:
            sim1 = fuzz.partial_ratio(ref, normalized_doc_text)
            sim2 = fuzz.partial_ratio(normalized_doc_text, ref)

            similarity = max(sim1, sim2)

            if similarity >= threshold:
                matched_docs.append(doc)
                break # break after first match per doc
    return matched_docs