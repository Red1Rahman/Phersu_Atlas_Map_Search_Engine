# map_api/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import QuerySerializer, RAGResponseSerializer
import logging

# Import the necessary Haystack components and pipeline setup
from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from haystack.dataclasses.chat_message import ChatMessage, ChatRole
import os # For GOOGLE_API_KEY environment variable

logger = logging.getLogger(__name__)

# --- RAG Pipeline Initialization (Global or Singleton Pattern) ---
rag_pipeline_instance = None

def get_rag_pipeline():
    """
    Initializes and returns the RAG pipeline.
    This function ensures the pipeline is warmed up only once.
    """
    global rag_pipeline_instance
    if rag_pipeline_instance is None:
        logger.info("Initializing RAG pipeline for the API...")
        
        CHROMA_SAVEPATH = "./data/chroma_db"
        ds = ChromaDocumentStore(persist_path=CHROMA_SAVEPATH)

        rag_pipeline = Pipeline()
        rag_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model="all-MiniLM-L6-v2"))
        rag_pipeline.add_component("retriever", ChromaEmbeddingRetriever(document_store=ds))
        rag_pipeline.add_component("prompt_builder", ChatPromptBuilder(template=[
            ChatMessage.from_system("""
                You are a helpful assistant extracting map-relevant information from documents.

                Your task is to find geographic locations mentioned in the context and describe their relevance. 
                Only include locations that are meaningful to the query or context.

                Return your output in the following strict JSON format:
                [
                {
                    "location_name": "name of the place",
                    "description": "why it is relevant or what is happening there"
                },
                ...
                ]

                Guidelines:
                - Be concise: keep descriptions under 30 words
                - Be accurate: do not fabricate location names
                - Be consistent: always return a JSON array (even if empty)
                - If no locations are relevant, return: []

                Context:
                {% for doc in documents %}
                {{ doc.content }}
                {% endfor %}
                """),
            ChatMessage.from_user("{{query}}")
        ],
        required_variables=['documents', 'query']
        ))
        rag_pipeline.add_component("gemini_generator", GoogleGenAIChatGenerator(model="gemini-1.5-flash"))

        rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder.prompt", "gemini_generator.messages")

        logger.info("Warming up the RAG pipeline components for API...")
        try:
            rag_pipeline.warm_up()
            logger.info("RAG pipeline components warmed up successfully for API.")
        except Exception as e:
            logger.error(f"ERROR: Failed to warm up RAG pipeline components for API: {e}", exc_info=True) # Added exc_info=True
            logger.error("Ensure GOOGLE_API_KEY is set in your environment (e.g., in the batch script or system-wide).")
            raise e # Re-raise to prevent server from starting with a broken pipeline

        rag_pipeline_instance = rag_pipeline
    return rag_pipeline_instance

class RAGQueryAPIView(APIView):
    """
    API endpoint to handle RAG queries.
    Expects a POST request with a 'query' field.
    """
    def post(self, request, *args, **kwargs):
        serializer = QuerySerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        user_query = serializer.validated_data['query']
        try:
            pipeline = get_rag_pipeline()
            results = pipeline.run({
                "text_embedder": {"text": user_query},
                "prompt_builder": {"query": user_query}
            })

            llm_reply = results.get("gemini_generator", {}).get("replies", ["No answer generated."])[0]
            raw_text = llm_reply.content[0].text if hasattr(llm_reply, "content") else llm_reply

            # ✅ JSON parsing
            try:
                structured_data = json.loads(raw_text)
                if not isinstance(structured_data, list):
                    structured_data = []
            except Exception as e:
                logger.warning(f"Failed to parse JSON from LLM: {e}")
                structured_data = []

            # ✅ Format retrieved documents
            retrieved_docs = results.get("retriever", {}).get("documents", [])
            retrieved_serialized = []
            for doc in retrieved_docs:
                snippet = doc.content[:200] + "..." if doc.content and len(doc.content) > 200 else doc.content
                retrieved_serialized.append({
                    "id": doc.id,
                    "score": doc.score,
                    "content_snippet": snippet
                })

            # ✅ Final response
            response_data = {
                "structured_locations": structured_data,
                "retrieved_documents": retrieved_serialized,
                "raw_llm_output": raw_text
            }

            response_serializer = RAGResponseSerializer(data=response_data)
            response_serializer.is_valid(raise_exception=True)
            return Response(response_serializer.data, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"RAG processing failed: {e}", exc_info=True)
            return Response(
                {"error": "Internal server error", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
