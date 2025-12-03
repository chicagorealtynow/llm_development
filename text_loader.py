from langchain_community.document_loaders import TextLoader

loader = TextLoader("greg.txt", encoding="utf-8")
documents = loader.load()

for doc in documents:
    print(f"Document Content: {doc.page_content}")
    print(f"Document Metadata: {doc.metadata}")
    print(f"Document Source: {doc.metadata['source']}")

    

