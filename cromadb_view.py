import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collections = client.list_collections()
print("Coleções disponíveis:", collections)

collection = client.get_collection(name="pdf_documents")

result = collection.get(
    include=["documents", "metadatas", "embeddings"]
)

print("\nTotal de documentos:", collection.count())
print("\nDocumentos:", result['documents'])
print("\nMetadados:", result['metadatas'])
print("\nIDs:", result['ids'])