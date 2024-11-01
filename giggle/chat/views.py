from rest_framework import views
from rest_framework.response import Response
from .services.chatbot_service import ChatBotService


class ChatBotAPIView(views.APIView):
    def get(self, request, *args, **kwargs):
        response = ChatBotService().get_user_history_message(request)
        return Response({"success": True, "data": response, "error": None})

    def post(self, request, *args, **kwargs):
        prompt = request.data.get('prompt')
        response = ChatBotService().chat_message(request, prompt)
        return Response({"success": True, "data": response, "error": None})