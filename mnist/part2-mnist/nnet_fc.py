#! /usr/bin/env python

import _pickle as cPickle, gzip
import numpy as np
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append("..")
#import utils
from utils import get_MNIST_data
from train_utils import batchify_data, run_epoch, train_model
import pandas as pd

def prep_data():
    # Load the dataset
    X_train, y_train, X_test, y_test = get_MNIST_data()

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = y_train[dev_split_index:]
    X_train = X_train[:dev_split_index]
    y_train = y_train[:dev_split_index]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [y_train[i] for i in permutation]

    return X_train, y_train, X_dev, y_dev, X_test, y_test


def batch_data(X_train, y_train, X_dev, y_dev, X_test, y_test, batch_size=32):
    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)
    
    return train_batches, dev_batches, test_batches


def run_model(train_batches, dev_batches, test_batches, lr=0.1, momentum=0, act_fun='ReLU', hidden_size=10):
    torch.manual_seed(12321)  # for reproducibility

    act_func_call = nn.ReLU
    if act_fun == 'LeakyReLU':
        act_func_call = nn.LeakyReLU
        
    
    #################################
    ## Model specification TODO
    model = nn.Sequential(
              nn.Linear(784, hidden_size),
              act_func_call(),
              nn.Linear(hidden_size, 10),
            )
    ##################################

    val_acc = train_model(train_batches, dev_batches, model, lr=lr, momentum=momentum)

    ## Evaluate the model on test data
    loss, accuracy = run_epoch(test_batches, model.eval(), None)
    
    return val_acc, loss, accuracy



if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    
    # create a set of model parameters
    model_params = pd.DataFrame({
        'Test Case' : ['Baseline', 'Batch Size', 'Learning Rate', 'Momentum', 'Activation'],
         'Batch Size' : [32, 64, 32, 32, 32],
         'Learning Rate' : [0.1, 0.1, 0.01, 0.1, 0.1],
         'Momentum' : [0, 0, 0, 0.9, 0],
         'Act Func' : ['ReLU', 'ReLU', 'ReLU', 'ReLU', 'LeakyReLU']}
    )

    X_train, y_train, X_dev, y_dev, X_test, y_test = prep_data()


    res = []
    for _,row in model_params.iterrows():
        train_batches, dev_batches, test_batches = batch_data(X_train, y_train, X_dev, y_dev, X_test, y_test, row['Batch Size'])
        val_acc, loss, accuracy = run_model(train_batches, dev_batches, test_batches, 
                                            row['Learning Rate'], row['Momentum'], row['Act Func'])

        res.append([val_acc, loss, accuracy])
        print ("Loss on test set:"  + str(loss) + " Accuracy on test set: " + str(accuracy))
        
    model_results = model_params.join(pd.DataFrame(res, columns=['Val Accuracy', 'Loss', 'Accuracy']))
    print(model_results)
