"""
URL configuration for chess_tutor project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from . import chat_views, practice_views, views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.chat_view, name='chat'),
    path('practice/', practice_views.practice_page, name='practice'),
    path('api/practice/', practice_views.practice_api, name='practice_api'),
    path('api/state/', views.state_view, name='state'),
    path('api/action/', views.action_view, name='action'),
    path('api/export/', views.export_view, name='export'),
    path('api/chat/', chat_views.chat_api, name='chat_api'),
    path('send_message/', chat_views.chat_api, name='send_message'),
]
