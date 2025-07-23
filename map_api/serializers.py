from rest_framework import serializers

class QuerySerializer(serializers.Serializer):
    query = serializers.CharField(min_length=5, max_length=150)

class DocumentMetadataSerializer(serializers.Serializer):
    id = serializers.CharField(max_length=255)
    score = serializers.FloatField()
    content_snippet = serializers.CharField()

class RAGResponseSerializer(serializers.Serializer):
    answer = serializers.CharField(help_text="Conversational reply from the LLM.")
    retrieved_documents = DocumentMetadataSerializer(many=True)
    full_document_contents = serializers.ListField(child=serializers.CharField(), required=False)

    structured_locations = serializers.ListField(child=serializers.DictField(), required=False)
    structured_time_periods = serializers.ListField(child=serializers.DictField(), required=False)
    structured_rulers_or_polities = serializers.ListField(child=serializers.DictField(), required=False)

    raw_llm_output = serializers.CharField(required=False)
