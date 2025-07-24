from django.db import models
from django.contrib.auth.models import User

class ChatMessageHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    role = models.CharField(max_length=10, choices=[('user', 'User'), ('assistant', 'Assistant')])
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    embedding = models.CharField(max_length=50, default="e5")
    structured_data = models.JSONField(null=True, blank=True)
    retrieved_documents = models.JSONField(null=True, blank=True)

    class Meta:
        ordering = ['timestamp']
