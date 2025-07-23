from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import QuerySerializer, RAGResponseSerializer
from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from haystack.dataclasses.chat_message import ChatMessage
import logging, json, re

logger = logging.getLogger(__name__)

retrieval_pipeline = None
generation_pipeline = None


def get_pipelines():
    global retrieval_pipeline, generation_pipeline

    if retrieval_pipeline is None:
        store = ChromaDocumentStore(persist_path="./data/chroma_db")
        retrieval_pipeline = Pipeline()
        retrieval_pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-mpnet-base-v2"))
        retrieval_pipeline.add_component("retriever", ChromaEmbeddingRetriever(document_store=store))
        retrieval_pipeline.connect("embedder.embedding", "retriever.query_embedding")
        retrieval_pipeline.warm_up()

    if generation_pipeline is None:
        generation_pipeline = Pipeline()
        generation_pipeline.add_component("generator", GoogleGenAIChatGenerator(model="gemini-1.5-flash"))
        generation_pipeline.warm_up()

    return retrieval_pipeline, generation_pipeline


class RAGQueryAPIView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = QuerySerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        query = serializer.validated_data['query']
        logger.info(f"Received query: {query}")
        try:
            retrieval_pipeline, generation_pipeline = get_pipelines()
            retrieval_results = retrieval_pipeline.run({"embedder": {"text": query}})
            retrieved_docs = retrieval_results["retriever"]["documents"]
            valid_docs = [doc for doc in retrieved_docs if getattr(doc, "content", None)]

            if not valid_docs:
                return Response({"answer": "No relevant documents found.", "retrieved_documents": [], "full_document_contents": [], "structured_locations": [], "structured_time_periods": [], "structured_rulers_or_polities": [], "raw_llm_output": ""})

            context = "\n\n".join(doc.content[:1000] for doc in valid_docs)
            messages = [
                ChatMessage.from_system("""
You are a helpful assistant who answers questions based on the provided historical context and returns structured data.

Answer the user's query using the context and then return structured data in JSON format:

Answer:
<your response>

Structured JSON:
{
  "locations": [{"name": "string", "description": "string"}],
  "time_periods": [{"name": "string", "description": "string"}],
  "rulers_or_polities": [{"name": "string", "description": "string"}]
}
                """),
                ChatMessage.from_user(f"Context:\n{context}\n\nUser Query: {query}")
            ]

            generation_result = generation_pipeline.run({"generator": {"messages": messages}})
            llm_output = generation_result["generator"]["replies"][0].text
            llm_output_cleaned = llm_output.replace("Answer: Answer:", "Answer:").strip()
            llm_output_cleaned = re.sub(r"(?s)```json(.*?)```", r"\1", llm_output_cleaned)

            match = re.search(r"Answer:\s*(.*?)\s*Structured JSON:\s*(\{.*)", llm_output_cleaned, re.DOTALL)
            if match:
                conversational_answer = match.group(1).strip()
                json_part = match.group(2).strip()
                try:
                    structured_json = json.loads(json_part)
                except Exception as e:
                    logger.warning(f"Failed to parse structured JSON: {e}")
                    structured_data = {}
            else:
                conversational_answer = llm_output.strip()
                structured_json = {}


            logger.info(f"Answer: {conversational_answer}")

            structured_data = {
                "structured_locations": structured_json.get("locations", []),
                "structured_time_periods": structured_json.get("time_periods", []),
                "structured_rulers_or_polities": structured_json.get("rulers_or_polities", [])
            }

            logger.info(f"Structured Data: {structured_json}")

            response_data = {
                "answer": conversational_answer,
                "retrieved_documents": [
                    {"id": doc.id, "score": doc.score} for doc in valid_docs
                ],
                "full_document_contents": [doc.content for doc in valid_docs],
                **structured_data,
                "raw_llm_output": llm_output
            }

            return Response(response_data)

        except Exception as e:
            logger.exception("RAG query failed")
            return Response({"error": str(e)}, status=500)
