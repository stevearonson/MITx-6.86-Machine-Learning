# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:47:47 2019

@author: 20448
"""

# from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import gzip, _pickle
from skorch import NeuralNetClassifier
from skorch import NeuralNetRegressor

device = 'cuda' if torch.cuda.is_available() else 'cpu'


img_rows, img_cols = 42, 28

def get_data(path_to_data_dir, use_mini_dataset):
    if use_mini_dataset:
        exten = '_mini'
    else:
        exten = ''
    f = gzip.open(path_to_data_dir + 'train_multi_digit' + exten + '.pkl.gz', 'rb')
    X_train = _pickle.load(f, encoding='latin1')
    f.close()
    X_train =  np.reshape(X_train, (len(X_train), 1, img_rows, img_cols))
    f = gzip.open(path_to_data_dir + 'test_multi_digit' + exten +'.pkl.gz', 'rb')
    X_test = _pickle.load(f, encoding='latin1')
    f.close()
    X_test =  np.reshape(X_test, (len(X_test),1, img_rows, img_cols))
    f = gzip.open(path_to_data_dir + 'train_labels' + exten +'.txt.gz', 'rb')
    y_train = np.loadtxt(f)
    f.close()
    f = gzip.open(path_to_data_dir +'test_labels' + exten + '.txt.gz', 'rb')
    y_test = np.loadtxt(f)
    f.close()
    return X_train, y_train, X_test, y_test

path_to_data_dir = '../Datasets/'
use_mini_dataset = True
X_train, y_train, X_test, y_test = get_data(path_to_data_dir, use_mini_dataset)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

y_train = y_train.astype('int64').T
y_test = y_test.astype('int64').T

#y_train = [tuple(x) for x in y_train.T]
#y_test = [tuple(x) for x in y_test.T]
#y_train = np.array(y_train,'i,i')
#y_test = np.array(y_test,'i,i')


def plot_example(X, y):
    """Plot the first 5 images and their labels in a row."""
    fig, axs = plt.subplots(1, 5, figsize=(12,4))
    for ix, ax in enumerate(axs):
        ax.imshow(X[ix].squeeze())
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('%d %d' % tuple(y[ix]))

plot_example(X_train, y_train)

class Flatten(nn.Module):
    """A custom layer that views an input as 1D."""
    
    def forward(self, input):
        return input.view(input.size(0), -1)


class CNN(nn.Module):

    def __init__(self):
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
        

torch.manual_seed(0)

class CNN_net(NeuralNetRegressor):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        
        loss1 = F.cross_entropy(y_pred[0], y_true[:,0])
        loss2 = F.cross_entropy(y_pred[1], y_true[:,1])
        
        return 0.5 * (loss1 + loss2)

net = CNN_net(
    CNN,
    max_epochs=5,
    lr=0.1,
    device=device,
)

net.fit(X_train, y_train);