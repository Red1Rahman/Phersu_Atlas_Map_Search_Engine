from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import QuerySerializer, RAGResponseSerializer
import logging

from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from haystack.dataclasses.chat_message import ChatMessage, ChatRole
import os

logger = logging.getLogger(__name__)

rag_pipeline_instance = None

def get_rag_pipeline():
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
        rag_pipeline.add_component("gemini_generator", GoogleGenAIChatGenerator(model="gemini-1.5-flash"))

        rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder.prompt", "gemini_generator.messages")

        logger.info("Warming up the RAG pipeline components for API...")
        try:
            rag_pipeline.warm_up()
            logger.info("RAG pipeline components warmed up successfully for API.")
        except Exception as e:
            logger.error(f"ERROR: Failed to warm up RAG pipeline components for API: {e}")
            logger.error("Ensure GOOGLE_API_KEY is set in your environment (e.g., in the batch script or system-wide).")
            # Re-raise the exception or handle it more gracefully in a production app
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
        logger.info(f"Received RAG query: '{user_query}'")

        try:
            pipeline = get_rag_pipeline()

            results = pipeline.run({
                "text_embedder": {"text": user_query},
                "prompt_builder": {"query": user_query}
            })

            llm_answer = results.get("gemini_generator", {}).get("replies", ["No answer generated."])[0]
            retrieved_docs_raw = results.get("retriever", {}).get("documents", [])

            retrieved_docs_serialized = []
            for doc in retrieved_docs_raw:
                doc_meta = {"id": doc.id, "score": doc.score}
                retrieved_docs_serialized.append(doc_meta)

            response_data = {
                "answer": llm_answer,
                "retrieved_documents": retrieved_docs_serialized
            }
            response_serializer = RAGResponseSerializer(data=response_data)
            response_serializer.is_valid(raise_exception=True)
            return Response(response_serializer.data, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error during RAG pipeline execution for query '{user_query}': {e}", exc_info=True)
            if "429 RESOURCE_EXHAUSTED" in str(e):
                return Response(
                    {"error": "API Quota Exceeded. Please try again later.", "details": str(e)},
                    status=status.HTTP_429_TOO_MANY_REQUESTS
                )
            return Response(
                {"error": "An internal server error occurred.", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

