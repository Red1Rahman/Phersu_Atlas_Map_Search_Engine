from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
import logging
import os
from datetime import datetime
from sentence_transformers import CrossEncoder

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup
CHHROMA_SAVEPATH = "./data/chroma_db"
timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")

# Chroma store
ds = ChromaDocumentStore(persist_path=CHHROMA_SAVEPATH, distance_function="cosine")

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_by_crossencoder(query, docs, top_k=5):
    pairs = [(query, doc.content) for doc in docs]
    scores = cross_encoder.predict(pairs)

    for doc, score in zip(docs, scores):
        doc.meta["cross_score"] = float(score)
    return sorted(docs, key=lambda d: d.meta["cross_score"], reverse=True)[:top_k]


# Components
embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-mpnet-base-v2")
retriever = ChromaEmbeddingRetriever(document_store=ds)

# Pipeline
pipeline = Pipeline()
pipeline.add_component("embedder", embedder)
pipeline.add_component("retriever", retriever)

pipeline.connect("embedder.embedding", "retriever.query_embedding")

# Warm-up
pipeline.warm_up()

# Run and write results
def run_queries(path="./data/input/questions.txt", top_k=10):
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
