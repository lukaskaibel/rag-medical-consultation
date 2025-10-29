from haystack import Pipeline
from pipelines.components.markdown_preprocessor import MarkdownPreprocessor

pipeline = Pipeline()

converter = MarkdownPreprocessor()

pipeline.add_component(instance=converter, name="markdown_converter")
