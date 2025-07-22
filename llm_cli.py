import os
import json
import logging
from datetime import datetime
from pathlib import Path
from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from haystack.dataclasses.chat_message import ChatMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_retrieval_pipeline():
    doc_store = ChromaDocumentStore(persist_path="./data/chroma_db")
    pipe = Pipeline()
    pipe.add_component("embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-mpnet-base-v2"))
    pipe.add_component("retriever", ChromaEmbeddingRetriever(document_store=doc_store))
    pipe.connect("embedder.embedding", "retriever.query_embedding")
    pipe.warm_up()
    return pipe

def build_generation_pipeline():
    pipe = Pipeline()
    pipe.add_component("generator", GoogleGenAIChatGenerator(model="gemini-1.5-flash"))
    return pipe

def main():
    retrieval_pipeline = build_retrieval_pipeline()
    generation_pipeline = build_generation_pipeline()

    print("Advanced Prompt CLI with Context-Aware JSON Extraction")
    print("--------------------------------------------------------")

    while True:
        query = input("Your question: ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        try:
            retrieval_results = retrieval_pipeline.run({
                "embedder": {"text": query}
            })
            retrieved_docs = retrieval_results["retriever"]["documents"]
            valid_docs = [doc for doc in retrieved_docs if getattr(doc, "content", "").strip()]

            if not valid_docs:
                print("⚠️ No documents with valid content retrieved. Skipping LLM call.")
                continue

            context = "\n\n".join(doc.content.strip()[:1000] for doc in valid_docs)

            messages = [
                ChatMessage.from_system("""
You are a historical assistant that extracts structured information from research documents.

Task:
- From the provided context and question, extract the following:
  1. Geographic locations (e.g., cities, regions, countries)
  2. Time periods (e.g., centuries, dynasties)
  3. Named rulers or polities (e.g., kings, empires)

Response Format:
Return only a JSON object like this:
{
  "locations": [{"name": "string", "description": "string"}],
  "time_periods": [{"name": "string", "description": "string"}],
  "rulers_or_polities": [{"name": "string", "description": "string"}]
}

Instructions:
- Be concise and informative.
- If an item isn't found, return an empty list for that field.
- Do NOT include explanations, markdown, or extra text.
"""),
                ChatMessage.from_user(f"Context:\n{context}\n\nUser Query: {query}")
            ]

            generation_results = generation_pipeline.run({
                "generator": {"messages": messages}
            })

            reply = generation_results["generator"]["replies"][0].text
            print("\n--- Gemini Response ---\n" + reply)

            # out_path = Path("output") / f"advanced_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            # out_path.parent.mkdir(exist_ok=True)
            # with open(out_path, "w") as f:
            #     json.dump({"query": query, "reply": reply}, f, indent=2)

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            print("Something went wrong. Check logs.")

if __name__ == "__main__":
    main()
