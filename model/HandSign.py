import glob
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms as T

class HandSign(Dataset):
    def __init__(self,img_size=224,transform=None):
        self.img_size = img_size
        self.imgs_path = "D:\stuff\\asl_to_speech\data\\asl_alphabet_train\\asl_alphabet_train"
        file_list = [f"{self.imgs_path}\\{i}" for i in os.listdir(self.imgs_path)]
        self.transform = transform
        # print(file_list)
        self.data = []
        self.class_map = {0: 'A',1: 'B',2: 'C',3: 'D',4: 'E',5: 'F',6: 'G',7: 'H',8: 'I',9: 'J',10: 'K'
                            ,11: 'L',12: 'M',13: 'N',14: 'O',15: 'P',16: 'Q',17: 'R',18: 'S',19: 'T',20: 'U'
                            ,21: 'V',22: 'W',23: 'X',24: 'Y',25: 'Z',26: 'del',27: 'space',28: 'nothing'}
        self.key_list = list(self.class_map.values())
       
        for class_path in file_list:
            class_name = class_path.split('\\')[-1]
            for img_path in os.listdir(class_path):
                self.data.append([os.path.join(self.imgs_path,class_name,img_path), class_name])
                # print(self.data[-1])
        
        self.img_dim = (self.img_size, self.img_size)
        # print(self.data[:5])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        if self.transform:
            img = self.transform(img)
        class_id = self.key_list.index(class_name)
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])

        return img_tensor, np.eye(29)[class_id]



if __name__ == "__main__":
    dataset = HandSign(img_size=224)
    print(len(dataset))
    data_loader = DataLoader(dataset,batch_size=8, shuffle=True)
    for i , (imgs, labels) in enumerate(data_loader):
        print("Batch of images has shape: ",imgs.shape,type(imgs))
        print("Batch of labels has shape: ", labels.shape,type(labels))
        
        if i >= 5:
            break