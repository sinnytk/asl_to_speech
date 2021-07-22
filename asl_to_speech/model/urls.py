from django.urls import path
from . import views
app_name = "model"
urlpatterns = [
    path("annotateImage", views.annotate_image, name="annotate_image"),
    path("annotateVideo", views.annotate_video, name="annotate_video"),

]
