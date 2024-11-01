from django.urls import path, include
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

urlpatterns = [
    path('v1/', include('chat.urls')),
]