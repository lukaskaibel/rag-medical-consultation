from rapidfuzz import fuzz
from utils.string_utils import normalize
from difflib import SequenceMatcher

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def _doc_matches(doc_tuple):
    """
    Worker helper: doc_tuple = (doc, normalized_doc_text, normalized_refs, threshold)
    Returns the original doc if it matches any ref, else None.
    """
    doc, text, refs, threshold = doc_tuple
    for ref in refs:
        if exact_substring_match_percent(ref, text) >= threshold:
            return doc
    return None

def find_matching_docs(references, documents, threshold=1.0, max_workers=None):
    """
    Returns all docs where exact_substring_match_percent ≥ threshold in either direction,
    checking documents in parallel.
    """
    # Pre-normalize once
    normalized_refs = [normalize(ref) for ref in references if isinstance(ref, str)]
    doc_tuples = [
        (doc, normalize(doc.content), normalized_refs, threshold)
        for doc in documents
    ]

    # By default use as many workers as you have cores
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    matched = []
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(_doc_matches, dt): dt[0] for dt in doc_tuples}
        for fut in as_completed(futures):
            doc = fut.result()
            if doc is not None:
                matched.append(doc)

    return matched


def exact_substring_match_percent(string1, string2):
    match = SequenceMatcher(None, string1, string2, autojunk=False).find_longest_match()
    max_matched_characters = match.size
    smaller_string_length = min(len(string1), len(string2))
    return max_matched_characters / smaller_string_length