from langchain_core.documents import Document

# Create a Document object
document = Document(
    page_content="This is the content of my document. It contains some important information.",
    metadata={
        "source": "my_local_file.txt",
        "author": "John Doe",
        "date": "2023-10-26",
        "category": "Technology"
    }
)

# Access the page_content and metadata
print(f"Document Content: {document.page_content}")
print(f"Document Metadata: {document.metadata}")

# Access specific metadata fields
print(f"Document Source: {document.metadata['source']}")