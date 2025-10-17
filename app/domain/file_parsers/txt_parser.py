from .base import BaseFileParser
from typing import List


class TxtParser(BaseFileParser):
    """Parser for extracting text chunks from TXT files."""

    def parse(self, file_path: str) -> List[str]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        # Split on double line-breaks, fallback to single lines if very short
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs or len(paragraphs) == 1:
            paragraphs = [line.strip() for line in text.split("\n") if line.strip()]
        return paragraphs
