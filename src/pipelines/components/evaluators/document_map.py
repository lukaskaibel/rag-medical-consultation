from typing import Any, Dict, List

from haystack import Document, component


@component
class DocumentMAPEvaluator:
    """
    A Mean Average Precision (MAP) evaluator for documents.

    Evaluator that calculates the mean average precision of the retrieved documents, a metric
    that measures how high retrieved documents are ranked.
    Each question can have multiple ground truth documents and multiple retrieved documents.

    `DocumentMAPEvaluator` doesn't normalize its inputs, the `DocumentCleaner` component
    should be used to clean and normalize the documents before passing them to this evaluator.

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.evaluators import DocumentMAPEvaluator

    evaluator = DocumentMAPEvaluator()
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
    # [1.0, 0.8333333333333333]
    print(result["score"])
    # 0.9166666666666666
    ```
    """

    @component.output_types(score=float, individual_scores=List[float])
    def run(
        self, ground_truth_documents: List[List[Document]], retrieved_documents: List[List[Document]]
    ) -> Dict[str, Any]:
        """
        Run the DocumentMAPEvaluator on the given inputs.

        All lists must have the same length.

        :param ground_truth_documents:
            A list of expected documents for each question.
        :param retrieved_documents:
            A list of retrieved documents for each question.
        :returns:
            A dictionary with the following outputs:
            - `score` - The average of calculated scores.
            - `individual_scores` - A list of numbers from 0.0 to 1.0 that represents how high retrieved documents
                are ranked.
        """
        if len(ground_truth_documents) != len(retrieved_documents):
            raise ValueError("The length of ground_truth_documents and retrieved_documents must be the same.")

        individual_scores: List[float] = []

        for ground_truth, retrieved in zip(ground_truth_documents, retrieved_documents):
            average_precision = 0.0
            average_precision_numerator = 0.0
            relevant_documents = 0

            # Collect ground truth IDs
            ground_truth_ids = {doc.id for doc in ground_truth if doc.id is not None}

            for rank, retrieved_document in enumerate(retrieved):
                if retrieved_document.id is None:
                    continue

                if retrieved_document.id in ground_truth_ids:
                    relevant_documents += 1
                    average_precision_numerator += relevant_documents / (rank + 1)

            if relevant_documents > 0:
                average_precision = average_precision_numerator / relevant_documents

            individual_scores.append(average_precision)

        score = sum(individual_scores) / len(individual_scores) if individual_scores else 0.0
        return {"score": score, "individual_scores": individual_scores}