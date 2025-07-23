from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from haystack.dataclasses.chat_message import ChatMessage, ChatRole
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_SAVEPATH = "./data/chroma_db_e5_embeddings"

def setup_retrieval_pipeline():
    logger.info(f"Initializing ChromaDocumentStore from: {CHROMA_SAVEPATH}")
    ds = ChromaDocumentStore(persist_path=CHROMA_SAVEPATH)

    try:
        doc_count = ds.count_documents()
        logger.info(f"Found {doc_count} documents in the ChromaDocumentStore.")
        if doc_count == 0:
            logger.warning("Warning: No documents found in the store. Please run ingestion first or check the data path.")
    except Exception as e:
        logger.error(f"Error checking document count: {e}")
        return None

    retrieval_pipeline = Pipeline()
    retrieval_pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model="intfloat/e5-large-v2"))
    retrieval_pipeline.add_component("retriever", ChromaEmbeddingRetriever(document_store=ds))
    retrieval_pipeline.add_component("prompt", ChatPromptBuilder(template=[
        ChatMessage.from_system("""
You are a helpful assistant that provides factual, concise answers grounded in the provided documents.

Documents:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}
"""),
        ChatMessage.from_user("{{query}}")
    ], required_variables=["documents", "query"]))
    retrieval_pipeline.add_component("generator", GoogleGenAIChatGenerator(model="gemini-1.5-flash"))

    retrieval_pipeline.connect("embedder.embedding", "retriever.query_embedding")
    retrieval_pipeline.connect("retriever.documents", "prompt.documents")
    retrieval_pipeline.connect("prompt.prompt", "generator.messages")

    logger.info("Warming up the LLM pipeline components...")
    try:
        retrieval_pipeline.warm_up()
        logger.info("LLM pipeline components warmed up successfully.")
    except Exception as e:
        logger.error(f"ERROR warming up LLM pipeline: {e}")
        return None

    return retrieval_pipeline

def main():
    pipeline = setup_retrieval_pipeline()
    if pipeline is None:
        print("Pipeline setup failed.")
        return

    print("\n🧠 Enter your question (or type 'exit' to quit):")
    while True:
        query = input(">>> ")
        if query.lower() in ("exit", "quit"):
            break

        try:
            results = pipeline.run({
                "embedder": {"text": query},
                "prompt": {"query": query}
            })

            answer = results.get("generator", {}).get("replies", ["No answer returned."])[0]
            print("\n🔍 Answer:\n", answer)

            docs = results.get("retriever", {}).get("documents", [])
            print("\n📄 Top documents used:")
            for doc in docs[:3]:
                print(f"- Score: {doc.score:.4f}")
                print(doc.content[:300] + ("..." if len(doc.content) > 300 else ""))
                print("-" * 60)

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            print("❌ Something went wrong. Check logs.")

if __name__ == "__main__":
    main()
