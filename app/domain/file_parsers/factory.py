from .pdf_parser import PDFParser
from .txt_parser import TxtParser
from .docx_parser import DocxParser
from .base import BaseFileParser
import os

PARSER_MAPPING = {
    ".pdf": PDFParser,
    ".txt": TxtParser,
    ".docx": DocxParser,
}


def get_parser_for_file(filename: str) -> BaseFileParser:
    ext = os.path.splitext(filename)[1].lower()
    parser_cls = PARSER_MAPPING.get(ext)
    if parser_cls is None:
        raise NotImplementedError(f"No parser available for extension: {ext}")
    return parser_cls()
