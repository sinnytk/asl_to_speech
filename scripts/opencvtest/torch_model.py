import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
label_mappings = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z',
    26: 'del',
    27: 'space',
    28: 'nothing'
}


class Net(nn.Module):
    def __init__(self, image_size = 100, kernel_size = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.image_size = image_size
        
        self.conv1 = nn.Conv2d(1, 32, self.kernel_size ) # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv2 = nn.Conv2d(32, 64, self.kernel_size ) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window
        self.norm1 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, self.kernel_size )
        self.conv4 = nn.Conv2d(128, 256, self.kernel_size )
        self.conv5 = nn.Conv2d(256, 512, self.kernel_size )
        self.norm2 = nn.BatchNorm2d(512)


        x = torch.randn(self.image_size,self.image_size).view(-1,1,self.image_size,self.image_size)
        self.eval()
        self._to_linear = None
        self.convs(x)

        
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(self._to_linear, 512) #flattening.
        self.fc2 = nn.Linear(512, 256) 
        self.norm1d = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256,len(label_mappings))

    def convs(self, x):
        # average pooling over 2x2
        x = F.avg_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.avg_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = self.norm1(x)
        x = F.avg_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.avg_pool2d(F.relu(self.conv4(x)), (2, 2))
        x = F.avg_pool2d(F.relu(self.conv5(x)), (2, 2))
        x = self.norm2(x)
        

        
        
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = self.dropout(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) # bc this is our output layer. No activation here.
        x = self.norm1d(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)


def load_model(model_path):
    model = Net(kernel_size=2)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    return model


def make_inference(model, input_image):
    if type(input_image) != np.ndarray:
        img = cv2.imdecode(np.frombuffer(
            input_image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    else:
        img = input_image
    img = cv2.cvtColor((img), cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100))/255.0
    cv2.imshow('inference_img', img)
    img = torch.Tensor(img).view(-1, 1, 100, 100)
    pred_mid = model(img)
    prediction = torch.argmax(pred_mid)
    label = label_mappings[prediction.item()]
    return label
