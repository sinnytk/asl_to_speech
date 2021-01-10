from django.urls import path
from . import views
app_name = "app"
urlpatterns = [
    path("", views.index),
    path("image/", views.image, name="image"),
    path("video/", views.video, name="video"),
    path("camera/", views.camera, name="camera"),


]
