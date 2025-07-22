from rest_framework import serializers

class QuerySerializer(serializers.Serializer):
    """
    Serializer for the incoming user query.
    """
    query = serializers.CharField(
        min_length=5,
        max_length=150,
        help_text="The question to ask the RAG pipeline."
    )

class DocumentMetadataSerializer(serializers.Serializer):
    """
    Serializer for the metadata of a retrieved document.
    """
    id = serializers.CharField(max_length=255)
    score = serializers.FloatField()
    content_snippet = serializers.CharField(help_text="A short excerpt from the retrieved document.")

class StructuredLocationSerializer(serializers.Serializer):
    """
    Serializer for structured location data extracted by Gemini.
    """
    location_name = serializers.CharField(help_text="Name of the extracted location.")
    description = serializers.CharField(help_text="Short explanation of its relevance or context.")

class RAGResponseSerializer(serializers.Serializer):
    """
    Serializer for the RAG pipeline's response including LLM output and structured data.
    """
    structured_locations = StructuredLocationSerializer(many=True, help_text="List of extracted locations with descriptions.")
    retrieved_documents = DocumentMetadataSerializer(many=True, help_text="List of retrieved document metadata.")
    raw_llm_output = serializers.CharField(help_text="The raw string output from the LLM.")
