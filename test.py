import time
import os
import torch
import argparse
import random
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from dataloaders.dataloaders import MNISTLoader, WineQualityLoader
from trainers.trainer_simple_mnist import SimpleMNISTTrainer
from trainers.trainer_mlpRegressor import MLPRegressorTrainer


if __name__ == '__main__':
    """
    things to be included in the argsparser:
    1) add other universal config params 
    2) experiment mode selections: 
        model, dataset, alg
    things not to be included in the argsparser:
    1) specific model/dataset/alg hyperparams
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, choices=['cuda','cpu'] , default='cuda')
    parser.add_argument('--dataset', type=str, choices=['tibs','mnist','cifar','pde'], default='mnist')
    parser.add_argument('--alg',type= str, choices=['naiveCP','featCP'],default='featCP')
    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = 'cuda:0'
        print('Using CUDA...')
    elif torch.backends.mps.is_available():
        args.device = 'mps'
        print('CUDA is not available. Using MPS (Metal Performance Shaders)...')
    else:
        args.device = 'cpu'
        print('CUDA and MPS are not available. Using CPU...')
    print(f'Using device: {args.device}')

    test_option = 'wine-quality'
    if test_option == 'mnist': # this is a demo of how i trained the MNIST net
        MNISTLoader = MNISTLoader(args)
        MNISTLoader.load_full_data()
        MNISTLoader.load_split_data(portion=0.05) # meaning i use 5% of the dataset to train and test
        SimpleMNISTTrainer = SimpleMNISTTrainer(args)
        SimpleMNISTTrainer.train_setup(MNISTLoader.split_trainloader)
        SimpleMNISTTrainer.train()
        SimpleMNISTTrainer.test_setup(MNISTLoader.split_testloader)
        SimpleMNISTTrainer.test()
    elif test_option == 'wine-quality':
        WineQualityLoader = WineQualityLoader(args)
        WineQualityLoader.load_full_data()
        WineQualityLoader.load_split_data(portion=0.5)
        MLPRegressorTrainer = MLPRegressorTrainer(args)
        MLPRegressorTrainer.train_setup()
        MLPRegressorTrainer.train()
        MLPRegressorTrainer.test_setup()
        MLPRegressorTrainer.test()
        # WineQualityLoader.get_data()

    
    