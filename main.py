import time
import os
import torch
import argparse
import random
import numpy as np
from dataloaders.dataloaders import MNISTLoader
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def main(args):
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
    
    # my_MNISTLoader = MNISTLoader(args)
    # my_MNISTLoader.load_data()
    # my_trainMNIST, my_testMNIST = my_MNISTLoader.get_data()

    return "success"

    
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
    
    main(args)