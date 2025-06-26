from typing import Any, Dict, List

from haystack import Document, component


@component
class DocumentMRREvaluator:
    """
    Evaluator that calculates the mean reciprocal rank of the retrieved documents.

    MRR measures how high the first retrieved document is ranked.
    Each question can have multiple ground truth documents and multiple retrieved documents.

    `DocumentMRREvaluator` doesn't normalize its inputs, the `DocumentCleaner` component
    should be used to clean and normalize the documents before passing them to this evaluator.

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.evaluators import DocumentMRREvaluator

    evaluator = DocumentMRREvaluator()
    result = evaluator.run(
        ground_truth_documents=[
            [Document(id="doc1")],
            [Document(id="doc2"), Document(id="doc3")],
        ],
        retrieved_documents=[
            [Document(id="doc1")],
            [Document(id="doc4"), Document(id="doc2"), Document(id="doc3")],
        ],
    )
    print(result["individual_scores"])
    # [1.0, 0.5]
    print(result["score"])
    # 0.75
    ```
    """

    # Refer to https://www.pinecone.io/learn/offline-evaluation/ for the algorithm.
    @component.output_types(score=float, individual_scores=List[float])
    def run(
        self, ground_truth_documents: List[List[Document]], retrieved_documents: List[List[Document]]
    ) -> Dict[str, Any]:
        """
        Run the DocumentMRREvaluator on the given inputs.

        `ground_truth_documents` and `retrieved_documents` must have the same length.

        :param ground_truth_documents:
            A list of expected documents for each question.
        :param retrieved_documents:
            A list of retrieved documents for each question.
        :returns:
            A dictionary with the following outputs:
            - `score` - The average of calculated scores.
            - `individual_scores` - A list of numbers from 0.0 to 1.0 that represents how high the first retrieved
                document is ranked.
        """
        if len(ground_truth_documents) != len(retrieved_documents):
            raise ValueError("The length of ground_truth_documents and retrieved_documents must be the same.")

        individual_scores: List[float] = []

        for ground_truth, retrieved in zip(ground_truth_documents, retrieved_documents):
            reciprocal_rank = 0.0

            # Collect ground truth IDs
            ground_truth_ids = {doc.id for doc in ground_truth if doc.id is not None}

            for rank, retrieved_document in enumerate(retrieved):
                if retrieved_document.id is None:
                    continue
                if retrieved_document.id in ground_truth_ids:
                    reciprocal_rank = 1.0 / (rank + 1)
                    break

            individual_scores.append(reciprocal_rank)

        score = sum(individual_scores) / len(individual_scores) if individual_scores else 0.0
        return {"score": score, "individual_scores": individual_scores}