import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from mappings import label_mappings
from utils import load, split

class Net(nn.Module):
    def __init__(self, image_size = 100, kernel_size = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.image_size = image_size
        
        self.conv1 = nn.Conv2d(1, 32, self.kernel_size ) # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv2 = nn.Conv2d(32, 64, self.kernel_size ) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window
        self.conv3 = nn.Conv2d(64, 128, self.kernel_size )
  

        x = torch.randn(self.image_size,self.image_size).view(-1,1,self.image_size,self.image_size)
        self._to_linear = None
        self.convs(x)

        
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self._to_linear, 512) #flattening.
        self.fc2 = nn.Linear(512, 256) 
        self.fc3 = nn.Linear(256,29)

    def convs(self, x):
        # average pooling over 2x2
        x = F.avg_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.avg_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.avg_pool2d(F.relu(self.conv3(x)), (2, 2))

        
        
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = self.dropout(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x)) # bc this is our output layer. No activation here.
        x = self.fc3(x)
        return F.softmax(x, dim=1)
    
    def acc_score(self, pred, actual):
        matches = [torch.argmax(i) == torch.argmax(j) for i,j in zip(pred, actual)]
        return matches.count(True)/len(matches)


    def _train(self, train_X, train_y, # training data
                    val_X, val_y, # validation data
                    train_batch_size = 128, val_batch_size = 128, # batch sizes
                    num_of_epochs = 10, # epochs
                    optimizer = optim.Adam, loss_function = nn.BCELoss, # extra functions
                    learning_rate = 0.001):

        results = {
            'train_accuracies' : [],
            'train_losses': [],
            'val_accuracies': [],
            'val_losses': []
        }

        # move to GPU if available, otherwise use CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'On device: {torch.cuda.get_device_name(0)}')    
        self.to(device)

        _optimizer = optimizer(self.parameters(), lr=learning_rate)
        _loss_function = loss_function() 

        t_progress_bar = tqdm(range(num_of_epochs))
        for epoch in t_progress_bar:
            acc = 0.0
            loss = 0.0
            for i in range(0, len(train_X),train_batch_size):
                batch_X = train_X[i:i+train_batch_size].view(-1,1,
                                                                self.image_size,self.image_size)
                batch_y = train_y[i:i+train_batch_size]

                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                self.zero_grad()
                
                outputs = self(batch_X)
                
                batch_loss = _loss_function(outputs, batch_y)
                batch_loss.backward()    
                _optimizer.step()
                
                acc += self.acc_score(outputs, batch_y)
                loss += batch_loss.item()

                
            acc = acc/(len(train_X)/train_batch_size)
            loss = loss/(len(train_X)/train_batch_size)

            with torch.no_grad():
                rand_split = random.randint(0,len(val_X)-val_batch_size)
                val_batch = val_X[rand_split:rand_split+val_batch_size].to(device)
                val_output = self(val_batch)
                val_acc = self.acc_score(val_output, val_y[rand_split:rand_split + val_batch_size])
                val_loss = _loss_function(val_output, val_y[rand_split:rand_split + val_batch_size].to(device)).item()
                
            results['train_accuracies'].append(acc)
            results['train_losses'].append(loss)
            results['val_accuracies'].append(val_acc)
            results['val_losses'].append(val_loss)

            t_progress_bar.set_description(f'EPOCH: {epoch},TRAIN_ACC: {round(val_acc,4)},TRAIN_LOSS: {round(loss,4)},VAL_ACC: {val_acc},VAL_LOSS: {val_loss}')
        return results

def create_config():
    config = {}
    parser = argparse.ArgumentParser()
    parser.add_argument('IMAGE_SIZE', type=int, help="Dimension (single) of the square image. ")
    parser.add_argument('TRAINING_DATA_PATH', type=str, help="Path of generated training data (numpy array)")
    args = parser.parse_args()

    config['IMAGE_SIZE'] = args.IMAGE_SIZE
    config['TRAINING_DATA_PATH'] = args.TRAINING_DATA_PATH
    return config


def main():
    config = create_config()

    # init the model
    net = Net(kernel_size=3)
    torch.cuda.empty_cache() 
    

    # load the generated data into tensors
    print('Loading data...')
    X, y = load(path = config['TRAINING_DATA_PATH'], image_size = config['IMAGE_SIZE'])

    train_X, train_y, test_X, test_y = split(X,y, split = 0.8)

    print('Data loaded...')

    # Training
    print('Start training')
    history = net._train(train_X, train_y, test_X, test_y, num_of_epochs=15)
    print('Trained!')

    fig, axs = plt.subplots(2)
    fig.suptitle('Training performance')
    axs[0].plot(history['train_accuracies'], label='Training Accuracy')
    axs[0].plot(history['val_accuracies'], label='Validation Accuracy')
    
    axs[0].set_xlabel("No. of EPOCHS")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend()

    axs[1].plot(history['train_losses'], label='Training Loss')
    axs[1].plot(history['val_losses'], label='Validation Loss')
    
    axs[1].set_xlabel("No. of EPOCHS")
    axs[1].set_ylabel("Loss")

    axs[1].legend()
    plt.show()

if __name__ == '__main__':
    main()
