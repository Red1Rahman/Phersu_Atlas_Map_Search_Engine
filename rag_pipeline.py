from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from datetime import datetime

CHHROMA_SAVEPATH = "./data/chroma_db"
timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")

ds = ChromaDocumentStore(persist_path=CHHROMA_SAVEPATH)

try:
    doc_count = ds.count_documents()
    print(f"Found {doc_count} documents in the ChromaDocumentStore.")
    if doc_count == 0:
        print("Warning: No documents found in the store. Please ingestion first or check the data path.")
except Exception as e:
    print(f"Error checking document count: {e}")
    print("Ensure ChromaDB is correctly initialized and the data path exists and is accessible.")

def write_n_retrieved_docs_to_file(retrieval_pipeline, timestamp, n):
    QUESTIONS = "./data/input/questions.txt"
    test_file_name = f"./data/output/qaRagTest_{timestamp}.txt"
    try:
        with open(QUESTIONS, "r") as file:
            lines = file.readlines()
            for line in lines:
                test_query = line.strip()
                # print(f"\nRunning retrieval for query: '{test_query}'")
                try:
                    results = retrieval_pipeline.run({"text_embedder": {"text": test_query}})
                    retrieved_docs = results["retriever"]["documents"]
                    if retrieved_docs:
                        # print(f"\n--- Retrieved {len(retrieved_docs)} documents for query: '{test_query}' ---")
                        # test_file_name = f"./data/output/{"".join(x for x in test_query if x.isalnum())}_{timestamp_str}.txt"
                        with open(test_file_name, 'a', encoding='utf-8') as f:
                            f.write(test_query  + "\n")
                            for i, doc in enumerate(retrieved_docs[:n]):
                                f.write(f"\nDocument {i+1} (ID: {doc.id}, Score: {doc.score:.4f}):")
                                f.write("Content:")
                                f.write(doc.content)
                                f.write("Metadata:")
                                f.write(str(doc.meta))
                                f.write("\n" + "-" * 50 + "\n")
                    else:
                        print(f"\nNo documents retrieved for query: '{test_query}'.")
                        print("Consider checking your indexed documents, the query, or the embedding model.")

                except Exception as e:
                    print(f"An error occurred during retrieval: {e}")
                    print("Please ensure your ChromaDB is running and accessible, and the embedding model is downloaded.")
    except Exception as e:
        print(f"An error occurred while reading the questions file: {e}")
        print("Please ensure the file path is correct and the file is readable.")


retrieval_pipeline = Pipeline()
retrieval_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model="all-MiniLM-L6-v2"))
retrieval_pipeline.add_component("retriever", ChromaEmbeddingRetriever(document_store=ds))
retrieval_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

write_n_retrieved_docs_to_file(retrieval_pipeline, timestamp_str, 3)

