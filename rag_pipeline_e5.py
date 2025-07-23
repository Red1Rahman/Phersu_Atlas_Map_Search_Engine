from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
import logging
import os
from datetime import datetime

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup
CHROMA_SAVEPATH = "./data/chroma_db_e5_embeddings"
timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")

# Chroma store
ds = ChromaDocumentStore(persist_path=CHROMA_SAVEPATH, distance_function="cosine")


# Components
embedder = SentenceTransformersTextEmbedder(model="intfloat/e5-large-v2")
retriever = ChromaEmbeddingRetriever(document_store=ds)

# Pipeline
pipeline = Pipeline()
pipeline.add_component("embedder", embedder)
pipeline.add_component("retriever", retriever)

pipeline.connect("embedder.embedding", "retriever.query_embedding")

# Warm-up
pipeline.warm_up()

# Run and write results
def run_queries(path="./data/input/questions - all.txt", top_k=10):
    output_file = f"./data/output/chroma_retrieval_results_{timestamp_str}.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(path, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    with open(output_file, "w", encoding="utf-8") as out:
        for i, query in enumerate(queries, 1):
            logger.info(f"[{i}/{len(queries)}] Query: {query}")
            try:
                result = pipeline.run({"embedder": {"text": query}})
                documents = result["retriever"]["documents"]
                out.write(f"Query: {query}\n")
                for idx, doc in enumerate(documents[:top_k]):
                    out.write(f"\nDocument {idx+1} (Score: {doc.score:.4f})\n")
                    out.write(doc.content + "\n")
                    out.write(str(doc.meta) + "\n")
                    out.write("-" * 50 + "\n")
                out.write("\n")
            except Exception as e:
                logger.error(f"Query failed: {query} — {e}", exc_info=True)
                out.write(f"\nQuery failed: {query}\nError: {e}\n{'-'*50}\n")

    logger.info(f"Results written to {output_file}")

# Run
run_queries(top_k=5)
