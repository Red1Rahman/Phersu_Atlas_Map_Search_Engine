from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from haystack.dataclasses.chat_message import ChatMessage, ChatRole
import os
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
        with open(QUESTIONS, "r", encoding='utf-8') as file:
            lines = file.readlines()

        with open(test_file_name, 'a', encoding='utf-8') as f:
            for line_num, line in enumerate(lines):
                test_query = line.strip()
                if not test_query:
                    continue

                f.write(str(test_query) + "\n")
                f.write("--- LLM Generated Answer ---\n")

                try:
                    results = retrieval_pipeline.run({
                        "text_embedder": {"text": test_query},
                        "prompt_builder": {"query": test_query}
                    })

                    embedder_output = results.get("text_embedder", {})
                    if "embedding" in embedder_output and embedder_output["embedding"]:
                        print(f"  Text Embedder generated embedding (length: {len(embedder_output['embedding'])}).")
                    else:
                        print("  WARNING: Text Embedder did NOT generate an embedding.")
                    retrieved_docs = results.get("retriever", {}).get("documents", [])
                    print(f"  Retriever returned {len(retrieved_docs)} documents.")
                    if not retrieved_docs:
                        print("  WARNING: Retriever returned an empty list of documents.")
                    built_messages = results.get("prompt_builder", {}).get("prompt", [])
                    print(f"  PromptBuilder built {len(built_messages)} messages.")
                    if not built_messages:
                        print("  WARNING: PromptBuilder built an empty list of messages.")


                    llm_reply = results.get("gemini_generator", {}).get("replies", ["No answer generated."])[0]

                    f.write(llm_reply + "\n\n")
                    f.write(f"--- Retrieved Documents (Top {n}) ---\n")
                    if retrieved_docs:
                        for i, doc in enumerate(retrieved_docs[:n]):
                            f.write(f"\nDocument {i+1} (ID: {doc.id}, Score: {doc.score:.4f}):\n")
                            f.write("Content:\n")
                            f.write(doc.content + "\n")
                            f.write("Metadata:\n")
                            f.write(str(doc.meta) + "\n")
                            f.write("-" * 50 + "\n")
                    else:
                        f.write(f"\nNo documents retrieved for query: '{test_query}'.\n")
                        f.write("Consider checking your indexed documents, the query, or the embedding model.\n")
                        f.write("-" * 50 + "\n")
                    f.write("\n" * 2)

                except Exception as e:
                    error_msg = f"An error occurred during RAG pipeline execution for query '{test_query}': {e}"
                    print(f"Error: {error_msg}")
                    f.write(f"Error: {error_msg}\n\n")
                    f.write("\n" + "-" * 50 + "\n")
            print(f"\nAll results written to: {test_file_name}")

    except FileNotFoundError as e:
        print(f"Error: Questions file not found at '{QUESTIONS}'. Please create it. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during file operations or general setup: {e}")
        print("Please ensure the file paths are correct, permissions are set, and your Gemini API key is valid.")


retrieval_pipeline = Pipeline()
retrieval_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model="all-MiniLM-L6-v2"))
retrieval_pipeline.add_component("retriever", ChromaEmbeddingRetriever(document_store=ds))
retrieval_pipeline.add_component("prompt_builder", ChatPromptBuilder(template=[
    ChatMessage.from_system("""
Given the following information, answer the question concisely and accurately.
If the provided information does not contain the answer, state that the answer is not found in the provided context.

Documents:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}
"""),
    ChatMessage.from_user("{{query}}")
]))
retrieval_pipeline.add_component("gemini_generator", GoogleGenAIChatGenerator(model="gemini-1.5-flash"))

retrieval_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
retrieval_pipeline.connect("retriever.documents", "prompt_builder.documents")
retrieval_pipeline.connect("prompt_builder.prompt", "gemini_generator.messages")

# --- CRITICAL FIX: Warm up the pipeline components ---
# This ensures all models and heavy resources are loaded before running queries
print("Warming up the RAG pipeline components...")
try:
    retrieval_pipeline.warm_up()
    print("RAG pipeline components warmed up successfully.")
except Exception as e:
    print(f"ERROR: Failed to warm up RAG pipeline components: {e}")
    print("This often indicates issues with model downloads or API key setup.")
# --- END CRITICAL FIX ---

write_n_retrieved_docs_to_file(retrieval_pipeline, timestamp_str, 3)