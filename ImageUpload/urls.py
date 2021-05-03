from django.contrib import admin
from django.conf.urls import url, include
from django.urls import path
from . import views

urlpatterns = [
    url(r'^$', views.home,name='home'),
    url(r'imageprocess', views.imageprocess, name='imageprocess')
]
