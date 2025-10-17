from .base import BaseFileParser
from unstructured.partition.pdf import partition_pdf
from typing import List


class PDFParser(BaseFileParser):
    """Parser for extracting text chunks from PDF files."""

    def parse(self, file_path: str) -> List[str]:

        pdf_chunks = partition_pdf(
            filename=file_path,
            # infer_table_structure=True,                  # Extract tables
            strategy="hi_res",  # Mandatory to infer tables and extract tables
            # extract_image_block_types=["Image"],         # Add 'Table' to list to extract image of tables
            # image_output_dir_path=output_path,           # If None, image and tables will saved in base64
            # extract_image_block_to_payload=True,         # If true, will extract base64 for API usage
            chunking_strategy="by_title",  # or 'basic' / 'by_title'
            max_characters=800,  # limits how big each chunk can be. Default to 500
            combine_text_under_n_chars=400,  # prevents tiny chunks by merging them. Default to 0
            new_after_n_chars=800,  # ensures chunks are not too long even if no natural split is found.
        )
        # Example:
        # Step 1: max_characters

        # - Chunk 1: A (500) + B (400) = 900 ✅ (≤ 1000)

        # - Chunk 2: C (300) ✅

        # Step 2: combine_text_under_n_chars

        # - Chunk 2 = 300 → > any tiny threshold → nothing merges

        # Step 3: new_after_n_chars = 700

        # - Chunk 1 = 900 → > 700 → split → 700 + 200

        # - Chunk 2 = 300 → <700 → stay
        return [el.text.strip() for el in pdf_chunks if el.text.strip()]
