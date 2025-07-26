from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from django.contrib.auth.models import User

from .serializers import QuerySerializer
from .models import ChatMessageHistory

from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from haystack.dataclasses.chat_message import ChatMessage

import logging, json, re

logger = logging.getLogger(__name__)

retrieval_pipelines = {}
generation_pipeline = None

def get_retrieval_pipeline(embedding_type="e5"):
    if embedding_type not in retrieval_pipelines:
        config = settings.EMBEDDING_MODELS.get(embedding_type)
        if not config:
            raise ValueError("Invalid embedding type")
        store = ChromaDocumentStore(persist_path=config["path"])
        pipeline = Pipeline()
        pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model=config["name"]))
        pipeline.add_component("retriever", ChromaEmbeddingRetriever(document_store=store))
        pipeline.connect("embedder.embedding", "retriever.query_embedding")
        pipeline.warm_up()
        retrieval_pipelines[embedding_type] = pipeline
    return retrieval_pipelines[embedding_type]

def get_generation_pipeline():
    global generation_pipeline
    if generation_pipeline is None:
        generation_pipeline = Pipeline()
        generation_pipeline.add_component("generator", GoogleGenAIChatGenerator(model="gemini-1.5-flash"))
        generation_pipeline.warm_up()
    return generation_pipeline

class RAGQueryAPIView(APIView):
    def post(self, request, *args, **kwargs):
        if getattr(settings, "USE_MOCK_RAG_RESPONSE", True):
            mock_data = {
                "answer": "Mock answer: Rome was founded in 753 BC.",
                "retrieved_documents": [
                    {"score": 0.98, "file_path": "Legendary_Rome.pdf"}
                ],
                "full_document_contents": [
                    "This is a mock document about the founding of Rome by Romulus in 753 BC."
                ],
                "structured_locations": [
                    {"name": "Rome", "description": "Capital of ancient Rome, traditionally founded in 753 BC."}
                ],
                "structured_time_periods": [
                    {"name": "8th century BC", "description": "The era traditionally associated with the founding of Rome."}
                ],
                "structured_rulers_or_polities": [
                    {"name": "Romulus", "description": "First King of Rome, according to legend."}
                ],
                "raw_llm_output": "Answer: Rome was founded in 753 BC.\nStructured JSON: {...}"
            }
            return Response(mock_data)
        
        serializer = QuerySerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        query = serializer.validated_data["query"]
        embedding = serializer.validated_data.get("embedding", "e5")

        guest_user, _ = User.objects.get_or_create(username="guest")
        logger.info(f"Received query: {query} using embedding: {embedding}")

        try:
            generation_pipeline = get_generation_pipeline()

            history_entries = ChatMessageHistory.objects.filter(user=guest_user).order_by("-timestamp")[:6][::-1]
            qa_pairs = []
            for entry in history_entries:
                if entry.role == "user":
                    qa_pairs.append({"user": entry.content})
                elif entry.role == "assistant" and qa_pairs:
                    qa_pairs[-1]["assistant"] = entry.content
            qa_pairs = qa_pairs[-3:]

            history_summary = "\n".join([
                f"Q: {pair['user']}\nA: {pair.get('assistant', '')}" for pair in qa_pairs
            ])

            recent_questions = [pair["user"] for pair in qa_pairs]
            keyword_hint = ", ".join(recent_questions[-2:]) if recent_questions else ""
            retrieval_context = f"Previous Keywords: {keyword_hint}\nHistory Summary: {history_summary}\nCurrent: {query}" if history_summary else query

            retrieval_pipeline = get_retrieval_pipeline(embedding)
            retrieval_results = retrieval_pipeline.run({"embedder": {"text": retrieval_context}})
            retrieved_docs = retrieval_results["retriever"]["documents"]
            valid_docs = [doc for doc in retrieved_docs if getattr(doc, "content", None)]

            if not valid_docs:
                return Response({
                    "answer": "No relevant documents found.",
                    "retrieved_documents": [],
                    "full_document_contents": [],
                    "structured_locations": [],
                    "structured_time_periods": [],
                    "structured_rulers_or_polities": [],
                    "raw_llm_output": "",
                    "chat_history": []
                })

            context = "\n\n".join(doc.content[:1000] for doc in valid_docs)

            prompt_template = f"""
                You are a helpful assistant that answers questions using the provided documents. In addition to answering the user's query, extract structured metadata from the documents.

                ---

                Conversation Summary (if available):
                {history_summary if history_summary else 'None'}

                ---

                Documents:
                {context}

                ---

                Task:
                1. Provide a concise answer to the user query.
                2. Extract structured metadata in the following JSON format:

                Structured JSON:
                {{
                "locations": [{{"name": "...", "description": "..."}}],
                "time_periods": [{{"name": "...", "description": "..."}}],
                "rulers_or_polities": [{{"name": "...", "description": "..."}}]
                }}

                User Query:
                {query}

                Only return the answer followed by the structured JSON.
            """
            
            prompt_messages = [ChatMessage.from_system(prompt_template)]
            prompt_messages.append(ChatMessage.from_user(f"Context:\n{context}\n\nUser Query: {query}"))

            generation_result = generation_pipeline.run({"generator": {"messages": prompt_messages}})
            llm_output = generation_result["generator"]["replies"][0].text
            llm_output_cleaned = llm_output.replace("Answer: Answer:", "Answer:").strip()
            llm_output_cleaned = re.sub(r"(?s)```json(.*?)```", r"\1", llm_output_cleaned)

            structured_data = {
                "structured_locations": [],
                "structured_time_periods": [],
                "structured_rulers_or_polities": [],
            }

            match = re.search(r"Answer:\s*(.*?)\s*Structured JSON:\s*(\{.*)", llm_output_cleaned, re.DOTALL)
            if match:
                conversational_answer = match.group(1).strip()
                json_part = match.group(2).strip()
                try:
                    structured_json = json.loads(json_part)
                    structured_data = {
                        "structured_locations": structured_json.get("locations", []),
                        "structured_time_periods": structured_json.get("time_periods", []),
                        "structured_rulers_or_polities": structured_json.get("rulers_or_polities", [])
                    }
                except Exception as e:
                    logger.warning(f"Failed to parse structured JSON: {e}")
            else:
                conversational_answer = llm_output.strip()

            ChatMessageHistory.objects.create(user=guest_user, role="user", content=query, embedding=embedding)
            ChatMessageHistory.objects.create(
                user=guest_user,
                role="assistant",
                content=llm_output_cleaned,
                embedding=embedding,
                structured_data=structured_data,
                retrieved_documents=[
                    {"id": doc.id, "score": doc.score, "content": doc.content[:300]}
                    for doc in valid_docs
                ]
            )

            chat_display = []
            for pair in qa_pairs:
                chat_display.append({"role": "user", "content": pair["user"]})
                if "assistant" in pair:
                    chat_display.append({"role": "assistant", "content": pair["assistant"]})

            chat_display.append({"role": "user", "content": query})
            chat_display.append({"role": "assistant", "content": conversational_answer})

            response_data = {
                "answer": conversational_answer,
                "retrieved_documents": [
                    {"id": doc.id, "score": doc.score} for doc in valid_docs
                ],
                "full_document_contents": [doc.content for doc in valid_docs],
                **structured_data,
                "raw_llm_output": llm_output,
                "chat_history": chat_display
            }

            logger.info("RAG query successful")
            logger.info(f"Answer: {conversational_answer}")
            logger.info(f"Structured data: {structured_data}")

            return Response(response_data)

        except Exception as e:
            logger.exception("RAG query failed")
            return Response({"error": str(e)}, status=500)


class ClearChatAPIView(APIView):
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        guest_user, _ = User.objects.get_or_create(username="guest")
        ChatMessageHistory.objects.filter(user=guest_user).delete()
        return Response({"message": "Chat history cleared."})
