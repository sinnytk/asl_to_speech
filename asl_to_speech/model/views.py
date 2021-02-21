from django.shortcuts import render
from .apps import ModelConfig
from .utils.torch_model import make_inference
from django.http import JsonResponse


def index(request):
    return JsonResponse(data={"hi": "hello"})


def annotate_image(request):
    if request.method == "POST":
        img = request.FILES['image']
        label = make_inference(ModelConfig.model, img)
        return JsonResponse({"annotation": label})
