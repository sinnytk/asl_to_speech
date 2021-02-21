from django.shortcuts import render
from .apps import ModelConfig
from django.http import JsonResponse


def index(request):
    return JsonResponse(data={"hi": "hello"})
