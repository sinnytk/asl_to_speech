import torch
import numpy as np
from tqdm import tqdm



# function: load_and_split
# Arguments:
#   path: source numpy data (array)
#   image_size: dimension of the square image

# Output: 
#   Tuple of tensors: (X, y)
def load(path, image_size=100):
    try:
        training_data = np.load(path, allow_pickle=True)
    except PermissionError:
        print(f'Insufficient persmissions to access {path}')
        exit(1)

    except FileNotFoundError:
        print(f'{path} not found')
        exit(1)
    np.random.shuffle(training_data)

    X = torch.Tensor([i[0] for i in training_data]).view(-1,1,image_size,image_size)
    X.div(255.0)
    
    y = torch.Tensor([i[1] for i in training_data])

    return X, y

# function: split
# Arguments:
#    X: input
#    y: label
#   split: ratio of train to test split, 0.8 means 80% train, 20% test
# Output: 
#   Tuple of tensors: (train_X, train_y, test_X, test_y)
def split(X, y, split=0.8):

    split_size =  int(len(X)*(1-split))

    train_X = X[:-split_size]
    print(f"len of train_x:{len(train_X)}")
    train_y = y[:-split_size]
    print(f"len of train_y:{len(train_y)}")
    test_X = X[-split_size:]
    print(f"len of test_x:{len(test_X)}")
    test_y = y[-split_size:]
    print(f"len of test_y:{len(test_y)}")

    return train_X, train_y, test_X, test_y