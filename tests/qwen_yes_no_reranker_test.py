import pytest
import torch
from haystack import Document
from haystack.utils import ComponentDevice, Device
from src.pipelines.components.qwen_yes_no_reranker import QwenYesNoReranker
import os

os.environ["HF_HUB_CACHE"] = "./model-assets/hugging-face"

@pytest.fixture(scope="module")
def reranker():
    # Instantiate and warm up once for all tests in this module
    r = QwenYesNoReranker(device=ComponentDevice.from_single(Device.gpu(id=1)))
    r.warm_up()
    # Move model to float32 on CPU if fp16 isn’t supported here
    if next(r.model.parameters()).device.type == "cpu" and r.model.dtype == torch.float16:
        r.model = r.model.to(dtype=torch.float32)
    return r



# Parameterized test for German capitals
@pytest.mark.parametrize("query, expected_content", [
    ("Was ist die Hauptstadt von Deutschland?", "Berlin ist die Hauptstadt von Deutschland."),
    ("Was ist die Hauptstadt von Österreich?", "Wien ist die Hauptstadt von Österreich."),
])
def test_reranker_german_ranking(reranker, query, expected_content):
    docs = [
        Document(content="Paris ist die Hauptstadt von Frankreich."),
        Document(content="Berlin ist die Hauptstadt von Deutschland."),
        Document(content="Wien ist die Hauptstadt von Österreich."),
        Document(content="Madrid ist die Hauptstadt von Spanien."),
        Document(content="Rom ist die Hauptstadt von Italien."),
    ]
    result = reranker.run(query=query, documents=docs, top_k=len(docs))
    ranked_docs = result["documents"]

    # The top-ranked document should match the expected content
    assert ranked_docs[0].content == expected_content

    # Its score must be strictly higher than all others
    top_score = ranked_docs[0].score
    other_scores = [d.score for d in ranked_docs[1:]]
    assert all(top_score > s for s in other_scores)

    # The top score should reflect a "yes" judgment
    assert top_score > 0.5