import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U
import pandas as pd

path_to_data_dir = '../Datasets/'
use_mini_dataset = True

batch_size = 64
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions
# input_dimension = img_rows * img_cols



class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.flatten = Flatten()
        self.fc1 = nn.Linear(2880, 64)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop1(x)
        out_first_digit = self.fc2(x)
        out_second_digit = self.fc3(x)

        return out_first_digit, out_second_digit

class CNN_1C(nn.Module):
    '''
    
    Simple one conv level model
    '''
    
    def __init__(self, conv_size=3, dropout_rate=0.5, act_func='ReLU'):
        super(CNN_1C, self).__init__()
        flatten_size = ((img_rows - conv_size + 1)//2) * ((img_cols - conv_size + 1)//2) * 32
        
        self.act_func_call = F.relu
        if act_func == 'LeakyReLU':
            self.act_func_call = F.leaky_relu
        
        self.conv1 = nn.Conv2d(1, 32, conv_size)
        self.flatten = Flatten()
        self.fc1 = nn.Linear(flatten_size, 64)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(self.act_func_call(self.conv1(x)), 2)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop1(x)
        out_first_digit = self.fc2(x)
        out_second_digit = self.fc3(x)

        return out_first_digit, out_second_digit
    
    
class CNN_aug(nn.Module):

    def __init__(self):
        super(CNN_aug, self).__init__()
        
        self.conv1a = nn.Conv2d(1, 32, 3)
        self.conv1b = nn.Conv2d(32, 32, 3)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2a = nn.Conv2d(32, 64, 3)
        self.conv2b = nn.Conv2d(64, 64, 3)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.flatten = Flatten()
        self.fc1 = nn.Linear(1792, 128)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.leaky_relu(self.conv1a(x))
        x = F.leaky_relu(self.conv1b(x))
        x = self.pool1(x)
        x = F.leaky_relu(self.conv2a(x))
        x = F.leaky_relu(self.conv2b(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop1(x)
        out_first_digit = self.fc2(x)
        out_second_digit = self.fc3(x)

        return out_first_digit, out_second_digit
    
    
def prep_data(path_to_data_dir, use_mini_dataset):
    '''
    
    Load the dataset
    '''

    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    return X_train, y_train, X_dev, y_dev, X_test, y_test


def batch_data(X_train, y_train, X_dev, y_dev, X_test, y_test, batch_size=32):
    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)
    
    return train_batches, dev_batches, test_batches

def run_model(train_batches, dev_batches, test_batches, conv_size, dropout_rate, act_func):
    '''
    
    '''
    torch.manual_seed(12321)  # for reproducibility
    model = CNN_1C(conv_size, dropout_rate, act_func)

    # Train
    val_acc = train_model(train_batches, dev_batches, model, n_epochs=10)

    ## Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    
    return val_acc, loss, acc
    

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility

    X_train, y_train, X_dev, y_dev, X_test, y_test = prep_data(path_to_data_dir, use_mini_dataset)

    '''
    Sweep of convolution kernel size
    model_params = pd.DataFrame({
         'Batch Size' : [64, 64, 64],
         'Conv Size' : [3, 5, 7],
         'Dropout Rate' : [0.5, 0.5, 0.5],
         'Act Func' : ['ReLU', 'ReLU', 'ReLU']}
    )
    Slight improvement for kernel=7
    '''

    '''
    Sweep of batch size
    model_params = pd.DataFrame({
         'Batch Size' : [32, 64, 128],
         'Conv Size' : [7, 7, 7],
         'Dropout Rate' : [0.5, 0.5, 0.5],
         'Act Func' : ['ReLU', 'ReLU', 'ReLU']}
    )
    Slight improvement for batch_size=64
    '''

    '''
    Compare activation function
    '''
    
    model_params = pd.DataFrame({
         'Batch Size' : [64, 64],
         'Conv Size' : [7, 7],
         'Dropout Rate' : [0.5, 0.5],
         'Act Func' : ['ReLU', 'LeakyReLU']}
    )



    res = []
    for _,row in model_params.iterrows():
        train_batches, dev_batches, test_batches = batch_data(X_train, y_train, X_dev, y_dev, X_test, y_test, row['Batch Size'])
        val_acc, loss, accuracy = run_model(train_batches, dev_batches, test_batches,
                                            row['Conv Size'], row['Dropout Rate'], row['Act Func'])

        res.append(np.mean((val_acc, loss, accuracy), axis=1))
        print ("Loss on test set:"  + str(loss) + " Accuracy on test set: " + str(accuracy))
        
    model_results = model_params.join(pd.DataFrame(res, columns=['Val Accuracy', 'Loss', 'Accuracy']))
    print(model_results)

    # Load model
#    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], accuracy[0], loss[1], accuracy[1]))

