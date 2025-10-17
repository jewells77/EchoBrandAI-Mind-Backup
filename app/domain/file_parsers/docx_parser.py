from .base import BaseFileParser
from typing import List
from docx import Document


class DocxParser(BaseFileParser):
    """Parser for extracting text chunks from DOCX files."""

    def parse(self, file_path: str) -> List[str]:
        doc = Document(file_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return paragraphs
