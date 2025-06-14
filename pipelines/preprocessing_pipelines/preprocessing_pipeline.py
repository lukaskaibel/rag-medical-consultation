from haystack import Pipeline
from haystack.components.preprocessors import DocumentSplitter
from pipelines.components.filter import Filter

pipeline = Pipeline()

document_paragraph_splitter = DocumentSplitter(split_by="passage", split_length=1, split_overlap=0, language="de")
filter = Filter(condition=lambda doc: bool(doc.content.strip()))

pipeline.add_component(instance=document_paragraph_splitter, name="splitter")
pipeline.add_component(instance=filter, name="filter")

pipeline.connect("splitter.documents", "filter.documents")
