from rest_framework import views
from rest_framework.response import Response
from .services.chatbot_service import ChatBotService


class ChatBotAPIView(views.APIView):
    def get(self, request, *args, **kwargs):
        user_id = request.query_params.get('user_id')
        response = ChatBotService().get_user_history_message(user_id)
        return Response({"success": True, "data": response, "error": None})

    def post(self, request, *args, **kwargs):
        user_id = request.data.get('user_id')
        prompt = request.data.get('prompt')
        response = ChatBotService().chat_message(user_id, prompt)
        return Response({"success": True, "data": response, "error": None})