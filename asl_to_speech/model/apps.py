import os
from django.apps import AppConfig
from django.conf import settings
from .utils.torch_model import load_model


class ModelConfig(AppConfig):
    name = 'model'

    model_path = os.path.join(settings.TRAINED_MODELS, "initial_model.pt")
    model = load_model(model_path)
