from django.urls import path
from rest_framework.routers import SimpleRouter
from .views import ChatBotAPIView

router = SimpleRouter()

urlpatterns = [
    path('chatbot', ChatBotAPIView.as_view(), name='챗봇'),
]