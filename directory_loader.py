from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path=".",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()
# print(len(docs))
print(docs[0].page_content)