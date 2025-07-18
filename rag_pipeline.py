from haystack import Pipeline

from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever

CHHROMA_SAVEPATH = "./data/chroma_db"
print(f"Initializing ChromaDocumentStore from: {CHHROMA_SAVEPATH}")


ds = ChromaDocumentStore(persist_path=CHHROMA_SAVEPATH)

try:
    doc_count = ds.count_documents()
    print(f"Found {doc_count} documents in the ChromaDocumentStore.")
    if doc_count == 0:
        print("Warning: No documents found in the store. Please ingestion first or check the data path.")
except Exception as e:
    print(f"Error checking document count: {e}")
    print("Ensure ChromaDB is correctly initialized and the data path exists and is accessible.")

retrieval_pipeline = Pipeline()
retrieval_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model="all-MiniLM-L6-v2"))
retrieval_pipeline.add_component("retriever", ChromaEmbeddingRetriever(document_store=ds))
retrieval_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

test_query = "When was Rome founded?" # Adjust this based on your PDF content


try:
    results = retrieval_pipeline.run({"text_embedder": {"text": test_query}})

    retrieved_docs = results["retriever"]["documents"]

    if retrieved_docs:
        print(f"\n--- Retrieved {len(retrieved_docs)} documents for query: '{test_query}' ---")
        for i, doc in enumerate(retrieved_docs):
            print(f"\nDocument {i+1} (ID: {doc.id}, Score: {doc.score:.4f}):")
            print("Content:")
            # Print only a snippet for brevity, or full content if preferred
            print(doc.content[:500] + "..." if len(doc.content) > 500 else doc.content)
            print("Metadata:")
            print(doc.meta)
            print("-" * 50)
    else:
        print(f"\nNo documents retrieved for query: '{test_query}'.")
        print("Consider checking your indexed documents, the query, or the embedding model.")

except Exception as e:
    print(f"An error occurred during retrieval: {e}")
    print("Please ensure your ChromaDB is running and accessible, and the embedding model is downloaded.")
