from typing import Any, Dict, List
from haystack import component
from haystack.dataclasses import ByteStream, Document
from haystack.components.converters import MarkdownToDocument

@component
class MarkdownPreprocessor():
    @component.output_types(documents=List[Document])
    def run(self, byte_streams: List[ByteStream], id: str, meta: Dict[str, Any]):
        if (len(byte_streams) == 0):
            msg = "No Bytestreams provided"
            raise ValueError(msg)
        if (len(id) == 0):
            raise ValueError("No id provided")
        if (len(meta) == 0):
            raise ValueError("Missing meta informations (title, knowledge_base_id)")
        
        converter = MarkdownToDocument()
        all_docs: List[Document] = []

        for bs in byte_streams:
            # 1) Decode the bytes to text
            text = bs.data.decode("utf-8")
            # 2) Split into lines and filter out short headings
            lines = text.splitlines()
            # title = take first line that only has one # to start
            filtered = [
                line
                for line in lines
                if not line.startswith("#")
            ]
            # 3) Re-assemble and re-encode
            cleaned_text = "\n".join(filtered)
            cleaned_bs = ByteStream(data=cleaned_text.encode("utf-8"))
            # 4) Convert cleaned markdown to Documents
            result = converter.run(
                sources=[cleaned_bs], 
                meta=meta
            )
            document = result["documents"][0]
            document.id = id

            is_document_content_empty = not document.content.strip()
            if is_document_content_empty:
                continue
            all_docs.append(document)

        return {"documents": all_docs}