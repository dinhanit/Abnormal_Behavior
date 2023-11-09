from django.urls import path
from . import views

urlpatterns = [
    path('analyze1/', views.analyze_sentiment1, name='analyze_sentiment1'),
    path('', views.home_page, name='home_page'),
    path('members/', views.members, name='members'),
    path('feedback/', views.feeback, name='feedback'),
    path('feedback_analysis/', views.feedback_analysis, name='feedback_analysis'),
    path('chatbot/', views.chatbot_view, name='chatbot_view'),
    path('hisfeedback/', views.hisfeedback, name='hisfeedback'),
    path('hischatbot/', views.hischatbot, name='hischatbot'),
    path('exam/', views.exam, name='exam'),
    path('abnormal/', views.abnormal, name='abnormal'),
    path('verify_code/', views.verify_code, name='verify_code'),
]
