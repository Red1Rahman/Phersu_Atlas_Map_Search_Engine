from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
# Removed: from haystack.components.builders import ChatPromptBuilder
# Removed: from haystack.dataclasses.chat_message import ChatMessage, ChatRole
import os
from datetime import datetime
import logging

# Configure logging for this script
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHHROMA_SAVEPATH = "./data/chroma_db"
timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")

# Initialize ChromaDocumentStore, ensuring cosine distance is set
ds = ChromaDocumentStore(persist_path=CHHROMA_SAVEPATH, distance_function="cosine")

try:
    doc_count = ds.count_documents()
    logger.info(f"Found {doc_count} documents in the ChromaDocumentStore.")
    if doc_count == 0:
        logger.warning("Warning: No documents found in the store. Please run ingestion first or check the data path.")
except Exception as e:
    logger.error(f"Error checking document count: {e}")
    logger.error("Ensure ChromaDB is correctly initialized and the data path exists and is accessible.")

def write_retrieval_results_to_file(retrieval_pipeline, timestamp, n):
    """
    Runs queries from questions.txt through the retrieval pipeline
    and writes the retrieved documents to a file.
    """
    QUESTIONS = "./data/input/questions.txt"
    test_file_name = f"./data/output/pure_retrieval_test_results_{timestamp}.txt"

    output_dir = os.path.dirname(test_file_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(QUESTIONS, "r", encoding='utf-8') as file:
            lines = file.readlines()

        with open(test_file_name, 'a', encoding='utf-8') as f:
            for line_num, line in enumerate(lines):
                test_query = str(line.strip())
                if not test_query:
                    continue

                logger.info(f"[{line_num + 1}/{len(lines)}] Processing Query for Pure Retrieval Test: '{test_query}'")

                f.write(f"Query: {test_query}\n")
                f.write("--- Retrieved Documents (Top {}) ---\n".format(n))

                try:
                    results = retrieval_pipeline.run({
                        "text_embedder": {"text": test_query}
                    })

                    logger.info(f"  Pipeline run completed for query: '{test_query}'")
                    
                    retrieved_docs = results.get("retriever", {}).get("documents", [])
                    logger.info(f"  Retriever returned {len(retrieved_docs)} documents.")
                    if not retrieved_docs:
                        logger.warning("  WARNING: Retriever returned an empty list of documents.")
                    
                    if retrieved_docs:
                        for i, doc in enumerate(retrieved_docs[:n]):
                            content_snippet = doc.content
                            f.write(f"\nDocument {i+1} (ID: {doc.id}, Score: {doc.score:.4f}):\n")
                            f.write("Content Snippet:\n")
                            f.write(content_snippet + "\n")
                            f.write("Metadata:\n")
                            f.write(str(doc.meta) + "\n")
                            f.write("-" * 50 + "\n")
                    else:
                        f.write(f"\nNo documents retrieved for query: '{test_query}'.\n")
                        f.write("Consider checking your indexed documents, the query, or the embedding model.\n")
                        f.write("-" * 50 + "\n")
                    
                    f.write("\n" * 2)

                except Exception as e:
                    error_msg = f"An error occurred during retrieval pipeline execution for query '{test_query}': {e}"
                    logger.error(f"Error: {error_msg}", exc_info=True)
                    f.write(f"Error: {error_msg}\n\n")
                    f.write("\n" + "-" * 50 + "\n")
            logger.info(f"\nAll pure retrieval test results written to: {test_file_name}")

    except FileNotFoundError as e:
        logger.error(f"Error: Questions file not found at '{QUESTIONS}'. Please create it. Details: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during file operations or general setup: {e}", exc_info=True)


# Define the pure retrieval pipeline components
retrieval_pipeline = Pipeline()
retrieval_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model="all-MiniLM-L6-v2"))
retrieval_pipeline.add_component("retriever", ChromaEmbeddingRetriever(document_store=ds))
# PromptBuilder component is NOT added here

# Connect only the text_embedder to the retriever
retrieval_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

logger.info("Warming up the pure retrieval pipeline components...")
try:
    retrieval_pipeline.warm_up()
    logger.info("Pure retrieval pipeline components warmed up successfully.")
except Exception as e:
    logger.error(f"ERROR: Failed to warm up pure retrieval pipeline components: {e}", exc_info=True)
    logger.error("This often indicates issues with model downloads or API key setup.")

# Run the pure retrieval test
write_retrieval_results_to_file(retrieval_pipeline, timestamp_str, 3)
