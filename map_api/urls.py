from django.urls import path
from .views import RAGQueryAPIView, ClearChatAPIView

urlpatterns = [
    path('rag-query/', RAGQueryAPIView.as_view(), name='rag_query_api'),
    path('clear-chat/', ClearChatAPIView.as_view(), name='clear-chat'),
]