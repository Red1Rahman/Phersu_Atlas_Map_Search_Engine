from django.urls import path
from .views import RAGQueryAPIView

urlpatterns = [
    path('rag-query/', RAGQueryAPIView.as_view(), name='rag_query_api'),
]