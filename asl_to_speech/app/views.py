from django.http.response import JsonResponse
from django.shortcuts import render


def index(request):
    return render(request, "app/app.html", {})


def image(request):

    return render(request, "app/image_app.html", {})


def video(request):
    if request.method == "POST":
        return JsonResponse({"annotation": "Welcome to ASL To Speech, testing video"})
    return render(request, "app/video_app.html", {})


def camera(request):
    return render(request, "app/camera_app.html", {})
