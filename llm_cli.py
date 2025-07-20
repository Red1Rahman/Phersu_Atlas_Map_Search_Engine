from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from haystack.dataclasses.chat_message import ChatMessage, ChatRole
import os
import logging

# Configure logging to show more details, especially for Haystack components
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the path to your persistent ChromaDB store
CHHROMA_SAVEPATH = "./data/chroma_db"

def setup_retrieval_pipeline():
    logger.info(f"Initializing ChromaDocumentStore from: {CHHROMA_SAVEPATH}")
    ds = ChromaDocumentStore(persist_path=CHHROMA_SAVEPATH)

    try:
        doc_count = ds.count_documents()
        logger.info(f"Found {doc_count} documents in the ChromaDocumentStore.")
        if doc_count == 0:
            logger.warning("Warning: No documents found in the store. Please run ingestion first or check the data path.")
    except Exception as e:
        logger.error(f"Error checking document count: {e}")
        logger.error("Ensure ChromaDB is correctly initialized and the data path exists and is accessible.")
        return None

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
    ],
    required_variables=['documents', 'query']
    ))
    retrieval_pipeline.add_component("gemini_generator", GoogleGenAIChatGenerator(model="gemini-1.5-flash"))

    # Connect components
    retrieval_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    retrieval_pipeline.connect("retriever.documents", "prompt_builder.documents")
    retrieval_pipeline.connect("prompt_builder.prompt", "gemini_generator.messages")

    logger.info("Warming up the LLM pipeline components...")
    try:
        retrieval_pipeline.warm_up()
        logger.info("LLM pipeline components warmed up successfully.")
    except Exception as e:
        logger.error(f"ERROR: Failed to warm up LLM pipeline components: {e}")
        logger.error("This often indicates issues with model downloads or API key setup.")
        return None
    
    return retrieval_pipeline

def main():
    """
    Main function to run the interactive LLM CLI.
    """
    pipeline = setup_retrieval_pipeline()
    if pipeline is None:
        logger.error("Failed to set up LLM pipeline. Exiting.")
        return

    print("\n--- Interactive LLM CLI ---")
    print("Type your questions below. Type 'exit' to quit.")
    print("---------------------------\n")

    while True:
        user_query = input("Your question: ")
        if user_query.lower() == 'exit':
            print("Exiting LLM CLI. Goodbye!")
            break

        if not user_query.strip():
            print("Please enter a non-empty question.")
            continue

        try:
            # Run the pipeline
            results = pipeline.run({
                "text_embedder": {"text": user_query},
                "prompt_builder": {"query": user_query}
            })

            # Extract results
            llm_reply = results.get("gemini_generator", {}).get("replies", ["No answer generated."])[0]
            retrieved_docs = results.get("retriever", {}).get("documents", [])

            print("\n--- LLM Generated Answer ---")
            print(llm_reply)

            print("\n--- Retrieved Documents (Top 3) ---")
            if retrieved_docs:
                for i, doc in enumerate(retrieved_docs[:3]): # Display top 3 docs
                    print(f"Document {i+1} (ID: {doc.id}, Score: {doc.score:.4f}):")
                    print(f"Content: {doc.content[:200]}...") # Print first 200 chars
                    print(f"Metadata: {doc.meta}")
                    print("-" * 30)
            else:
                print("No documents retrieved for this query.")
            print("---------------------------\n")

        except Exception as e:
            logger.error(f"An error occurred during pipeline execution: {e}")
            logger.error("Please ensure your Gemini API key is valid and your quota has not been exceeded.")
            print("An error occurred. Please check the console for details and try again.")
            print("---------------------------\n")

if __name__ == "__main__":
    main()
