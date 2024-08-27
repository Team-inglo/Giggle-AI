from django.urls import path, include
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def preflight_options(request, *args, **kwargs):
    response = JsonResponse({"message": "CORS preflight options successful."})
    response['Access-Control-Allow-Origin'] = '*'
    response['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE, PATCH'
    response['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

urlpatterns = [
    path('api/v1/', include('chatbot.urls')),
    path('api/v1/<path:path>', preflight_options),
]