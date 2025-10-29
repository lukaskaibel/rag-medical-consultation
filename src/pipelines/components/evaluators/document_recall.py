from enum import Enum
from typing import Any, Dict, List, Union

from haystack import component, default_to_dict, logging
from haystack.dataclasses import Document

logger = logging.getLogger(__name__)


class RecallMode(Enum):
    """
    Enum for the mode to use for calculating the recall score.
    """

    # Score is based on whether any document is retrieved.
    SINGLE_HIT = "single_hit"
    # Score is based on how many documents were retrieved.
    MULTI_HIT = "multi_hit"

    def __str__(self):
        return self.value

    @staticmethod
    def from_str(string: str) -> "RecallMode":
        """
        Convert a string to a RecallMode enum.
        """
        enum_map = {e.value: e for e in RecallMode}
        mode = enum_map.get(string)
        if mode is None:
            msg = f"Unknown recall mode '{string}'. Supported modes are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return mode


@component
class DocumentRecallEvaluator:
    """
    Evaluator that calculates the Recall score for a list of documents.

    Returns both a list of scores for each question and the average.
    There can be multiple ground truth documents and multiple predicted documents as input.

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.evaluators import DocumentRecallEvaluator

    evaluator = DocumentRecallEvaluator(mode="multi_hit")
    result = evaluator.run(
        ground_truth_documents=[
            [Document(id="doc1")],
            [Document(id="doc2"), Document(id="doc3")],
        ],
        retrieved_documents=[
            [Document(id="doc4"), Document(id="doc1")],
            [Document(id="doc3")],
        ],
    )
    print(result["individual_scores"])
    # [1.0, 0.5]
    print(result["score"])
    # 0.75
    ```
    """

    def __init__(self, mode: Union[str, RecallMode] = RecallMode.MULTI_HIT):
        """
        Create a DocumentRecallEvaluator component.

        :param mode:
            Mode to use for calculating the recall score.
        """
        if isinstance(mode, str):
            mode = RecallMode.from_str(mode)

        mode_functions = {
            RecallMode.SINGLE_HIT: DocumentRecallEvaluator._recall_single_hit,
            RecallMode.MULTI_HIT: DocumentRecallEvaluator._recall_multi_hit,
        }
        self.mode_function = mode_functions[mode]
        self.mode = mode

    @staticmethod
    def _recall_single_hit(ground_truth_documents: List[Document], retrieved_documents: List[Document]) -> float:
        # Collect non-null IDs
        truth_ids = {doc.id for doc in ground_truth_documents if doc.id is not None}
        retrieval_ids = {doc.id for doc in retrieved_documents if doc.id is not None}
        # Check if any ground truth ID was retrieved
        return float(bool(truth_ids & retrieval_ids))

    @staticmethod
    def _recall_multi_hit(ground_truth_documents: List[Document], retrieved_documents: List[Document]) -> float:
        truth_ids = {doc.id for doc in ground_truth_documents if doc.id is not None}
        retrieval_ids = {doc.id for doc in retrieved_documents if doc.id is not None}
        retrieved_truths = truth_ids & retrieval_ids

        if not truth_ids:
            logger.warning(
                "There are no ground truth documents with valid IDs. Score will be set to 0."
            )
            return 0.0

        if not retrieval_ids:
            logger.warning(
                "There are no retrieved documents with valid IDs. Score will be set to 0."
            )
            return 0.0

        return len(retrieved_truths) / len(truth_ids)

    @component.output_types(score=float, individual_scores=List[float])
    def run(
        self, ground_truth_documents: List[List[Document]], retrieved_documents: List[List[Document]]
    ) -> Dict[str, Any]:
        """
        Run the DocumentRecallEvaluator on the given inputs.

        `ground_truth_documents` and `retrieved_documents` must have the same length.

        :param ground_truth_documents:
            A list of expected documents for each question.
        :param retrieved_documents:
            A list of retrieved documents for each question.
        :returns:
            A dictionary with:
            - `score` - The average recall score across all questions.
            - `individual_scores` - A list of recall scores per question.
        """
        if len(ground_truth_documents) != len(retrieved_documents):
            raise ValueError("The length of ground_truth_documents and retrieved_documents must be the same.")

        scores: List[float] = []
        for gt_docs, ret_docs in zip(ground_truth_documents, retrieved_documents):
            scores.append(self.mode_function(gt_docs, ret_docs))

        average_score = sum(scores) / len(scores) if scores else 0.0
        return {"score": average_score, "individual_scores": scores}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.
        """
        return default_to_dict(self, mode=str(self.mode))