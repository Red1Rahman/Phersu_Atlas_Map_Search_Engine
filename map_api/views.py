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
import spacy
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

retrieval_pipelines = {}
generation_pipeline = None

# Load once
nlp = spacy.load("en_core_web_sm")
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

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

def extract_keywords(texts, top_k=10):
    all_phrases = []
    for text in texts:
        doc = nlp(text)
        tokens = [token.text for token in doc if token.pos_ in {"PROPN", "NOUN", "VERB", "ADJ"} and not token.is_stop]
        all_phrases.extend(tokens)

    if not all_phrases:
        return []

    unique_phrases = list(set(all_phrases))
    phrase_embeddings = bert_model.encode(unique_phrases, convert_to_tensor=True)
    question_embedding = bert_model.encode(" ".join(texts), convert_to_tensor=True)

    scores = util.cos_sim(question_embedding, phrase_embeddings)[0]
    top_indices = scores.argsort(descending=True)[:top_k]
    return [unique_phrases[i] for i in top_indices]

class RAGQueryAPIView(APIView):
    def post(self, request, *args, **kwargs):
        if getattr(settings, "USE_MOCK_RAG_RESPONSE", False):
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
                "raw_llm_output": "Answer: Rome was founded in 753 BC."
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
            keywords = extract_keywords(recent_questions) if recent_questions else []
            keyword_hint = ", ".join(keywords)
            retrieval_context = f"Keyword Hints: {keyword_hint}\nConversation Summary: {history_summary}\nQuery: {query}" if keyword_hint or history_summary else query

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

            doc_texts = [doc.content[:1000] for doc in valid_docs]
            context = "\n---\n".join(doc_texts)

            prompt_template = f"""
                You are a helpful assistant that answers user questions using the provided documents and extracts structured metadata.

                Conversation Summary:
                {history_summary if history_summary else ''}

                Documents:
                {context}

                Task:
                Answer the following query concisely.

                Then provide three lists:
                - Locations: A list of locations mentioned, with short descriptions.
                - Time Periods: A list of historical time periods mentioned, with short descriptions.
                - Rulers or Polities: A list of historical rulers, governments, or kingdoms mentioned, with short descriptions.

                Query: {query}
            """

            prompt_messages = [ChatMessage.from_system(prompt_template)]
            prompt_messages.append(ChatMessage.from_user(f"Context:\n{context}\n\nUser Query: {query}"))

            generation_result = generation_pipeline.run({"generator": {"messages": prompt_messages}})
            llm_output = generation_result["generator"]["replies"][0].text.strip()

            structured_data = {
                "structured_locations": [],
                "structured_time_periods": [],
                "structured_rulers_or_polities": [],
            }

            # Extract using labeled sections instead of JSON
            logger.info(f"LLM output: {llm_output}")
            conversational_answer = llm_output.split("Locations:")[0].strip()
            conversational_answer = "\n".join([
                re.sub(r"[^\w\s.,:;!?()-]", "", line).strip()
                for line in conversational_answer.splitlines()
                if line.strip()
            ])

            def extract_section(label):
                pattern = rf"{label}:\s*(.*?)(?:\n\w+:|$)"
                match = re.search(pattern, llm_output, re.DOTALL | re.IGNORECASE)
                return match.group(1).strip() if match else ""

            def parse_bullets(text):
                lines = [line.strip("-*• ").strip() for line in text.splitlines() if line.strip()]
                parsed = []
                for line in lines:
                    if ":" in line:
                        name, desc = line.split(":", 1)
                        # Remove common markdown formatting
                        clean_name = re.sub(r"\*+", "", name).strip()
                        clean_desc = re.sub(r"\*+", "", desc).strip()
                        parsed.append({"name": clean_name, "description": clean_desc})
                return parsed


            structured_data["structured_locations"] = parse_bullets(extract_section("Locations"))
            structured_data["structured_time_periods"] = parse_bullets(extract_section("Time Periods"))
            structured_data["structured_rulers_or_polities"] = parse_bullets(extract_section("Rulers or Polities"))

            ChatMessageHistory.objects.create(user=guest_user, role="user", content=query, embedding=embedding)
            ChatMessageHistory.objects.create(
                user=guest_user,
                role="assistant",
                content=llm_output,
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