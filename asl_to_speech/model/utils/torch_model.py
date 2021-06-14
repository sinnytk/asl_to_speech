import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import json

from .label_mapping import label_mappings


class Net(nn.Module):
    def __init__(self):
        super().__init__()  # just run the init of parent class (nn.Module)
        # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv1 = nn.Conv2d(1, 32, 5)
        # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(100, 100).view(-1, 1, 100, 100)
        self._to_linear = None
        self.convs(x)

        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(self._to_linear, 512)  # flattening.
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 29)

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        # .view is reshape ... this flattens X before
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # bc this is our output layer. No activation here.
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


def load_model(model_path):
    model = Net()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    return model


def make_inference(model, input_image):
    img = cv2.imdecode(np.frombuffer(
        input_image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 100))/255
    img = torch.Tensor(img).view(-1, 1, 100, 100)
    prediction = torch.argmax(model(img))
    label = label_mappings[prediction.item()]
    return label
