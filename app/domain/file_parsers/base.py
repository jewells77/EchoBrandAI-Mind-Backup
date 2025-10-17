from abc import ABC, abstractmethod
from typing import List


class BaseFileParser(ABC):
    """Abstract base class for file parsers."""

    @abstractmethod
    def parse(self, file_path: str) -> List[str]:
        """
        Parse the input file and return a list of text chunks.
        """
        pass
