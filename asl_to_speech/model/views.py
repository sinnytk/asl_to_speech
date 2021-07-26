from .apps import ModelConfig
from .utils.torch_model import image_to_inference, video_to_inference
from django.http import JsonResponse


def annotate_image(request):
    if request.method == "POST":
        img = request.FILES['image']
        label = image_to_inference(ModelConfig.model, img)
        return JsonResponse({"annotation": label})
