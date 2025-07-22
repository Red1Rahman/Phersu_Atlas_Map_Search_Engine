import os
import json
import logging
from datetime import datetime
from pathlib import Path

from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from haystack.dataclasses.chat_message import ChatMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_SAVEPATH = "./data/chroma_db"

def setup_pipeline():
    logger.info("Initializing ChromaDocumentStore...")
    ds = ChromaDocumentStore(persist_path=CHROMA_SAVEPATH)

    pipe = Pipeline()
    pipe.add_component("embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-mpnet-base-v2"))
    pipe.add_component("retriever", ChromaEmbeddingRetriever(document_store=ds))
    pipe.add_component("prompt_builder", ChatPromptBuilder(template=[
        ChatMessage.from_system("""
You are a helpful assistant. From the provided documents, extract any geographic or map-relevant locations.

Return your response as a JSON array of objects:
[
  {
    "location_name": "name of the place",
    "description": "why it is relevant or what is happening there"
  },
  ...
]

Guidelines:
- Only return the JSON — no explanation or commentary.
- If no relevant locations are found, return: []
Documents:
{% for doc in documents %}
  {{ doc.content }}
{% endfor %}
"""),
        ChatMessage.from_user("Query: {{query}}")
    ], required_variables=["documents", "query"]))
    pipe.add_component("generator", GoogleGenAIChatGenerator(model="gemini-1.5-flash"))

    pipe.connect("embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "prompt_builder.documents")
    pipe.connect("prompt_builder.prompt", "generator.messages")

    pipe.warm_up()
    return pipe

def main():
    pipeline = setup_pipeline()
    print("Ask your question (or type 'exit'):")

    while True:
        query = input("Your question: ").strip()
        if query.lower() == "exit":
            print("Exiting.")
            break
        if not query:
            print("Please enter a valid query.")
            continue

        try:
            results = pipeline.run({
                "embedder": {"text": query},
                "prompt_builder": {"query": query}
            })

            llm_reply = results.get("generator", {}).get("replies", ["No answer"])[0]
            raw_text = llm_reply.text if hasattr(llm_reply, "text") else str(llm_reply)

            print("\n--- LLM Answer ---")
            print(raw_text)

            print("\n--- Retrieved Documents ---")
            retrieved_docs = results.get("retriever", {}).get("documents", [])
            for i, doc in enumerate(retrieved_docs[:3]):
                print(f"\nDocument {i+1} (Score: {doc.score:.4f})")
                print(f"Snippet: {doc.content[:300].strip()}...")
                print(f"Meta: {doc.meta}")

            print("\n--- Parsed Map Data (if any) ---")
            try:
                structured_data = json.loads(raw_text)
                if isinstance(structured_data, list) and structured_data:
                    for i, item in enumerate(structured_data, 1):
                        print(f"{i}. {item.get('location_name', 'N/A')}: {item.get('description', 'No description')}")
                else:
                    print("No location data found.")
            except Exception as e:
                logger.warning(f"Failed to parse structured JSON: {e}")
                print("Fallback: displaying raw LLM response")
                print(raw_text)

            # Save result
            out_path = Path("output") / f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                try:
                    json.dump(structured_data, f, indent=2)
                except Exception:
                    f.write(raw_text)
            print(f"Results saved to {out_path}")

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            print("Something went wrong. Check logs.")

if __name__ == "__main__":
    main()
