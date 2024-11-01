from django.db import models

class ChatMessage(models.Model):

    user_id = models.CharField(max_length=255)
    role = models.CharField(max_length=20)
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

