from typing import List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader as BaseTextLoader

class TextLoader(BaseTextLoader):
    """Load text files."""

    def __init__(self, file_key: str, encoding: Optional[str] = None):
        """Initialize with file key."""
        self.file_key = file_key
        self.encoding = encoding

        # Split the file key back into the repository URL and relative file path
        repo_url, relative_file_path = file_key.rsplit('/', 1)

        # Pass the relative file path to the base class's constructor
        super().__init__(relative_file_path, encoding)

    def load(self) -> List[Document]:
        """Load from file path."""
        with open(self.file_path, encoding=self.encoding) as f:
            text = f.read()
        metadata = {"source": self.file_key}  # Use the file key as the source
        return [Document(page_content=text, metadata=metadata)]
