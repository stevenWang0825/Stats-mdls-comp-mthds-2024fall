import torch
import torchvision
import torchvision.transforms as transforms
import os
import sys
from ucimlrepo import fetch_ucirepo 
import random
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from baseclasses.baseclass import BaseDataLoader
from torch.utils.data import DataLoader, RandomSampler

class MNISTLoader(BaseDataLoader):
    def __init__(self, args) -> None:
        super(MNISTLoader, self).__init__(args)
    
    def load_full_data(self) -> None:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        # check if data is already downloaded
        if not os.path.exists('./datasets/MNIST'):
            print("Downloading MNIST dataset...")
            download = True
        else:
            print("MNIST dataset already downloaded.")
            download = False
        # load data
        self.full_trainset = torchvision.datasets.MNIST(root='./datasets', train=True, download=download, transform=transform)
        self.full_testset = torchvision.datasets.MNIST(root='./datasets', train=False, download=download, transform=transform)
        self.full_trainloader = DataLoader(self.full_trainset, batch_size=64, shuffle=True)
        self.full_testloader = DataLoader(self.full_testset, batch_size=64, shuffle=False)
    
    def load_split_data(self, portion) -> None:
        """
        use total-size*portion of the train set and test set of MNIST, randomly sampled
        """
        if portion >= 1 or portion <= 0:
            raise ValueError("Invalid! portion should be in (0,1)")
        split_n_train = int(portion * len(self.full_trainset))
        split_n_test = int(portion * len(self.full_testset))
        train_sampler = RandomSampler(self.full_trainset,num_samples=split_n_train,replacement=False)
        test_sampler = RandomSampler(self.full_testset,num_samples=split_n_test,replacement=False)
        self.split_trainloader = DataLoader(self.full_trainset, batch_size=64, sampler=train_sampler)
        self.split_testloader = DataLoader(self.full_testset, batch_size=64, sampler=test_sampler)

    def get_data(self, option):
        if option == 'full':
            return self.full_trainloader, self.full_testloader
        elif option == 'split':
            return self.split_trainloader, self.split_testloader
        else:
            raise NotImplementedError

class WineQualityLoader(BaseDataLoader):
    def __init__(self, args) -> None:
        super(WineQualityLoader, self).__init__(args)
    
    def load_full_data(self) -> None:
        """
        load the full uci wine quality dataset
        self.X: features matrix torch.tensor (6497,11)
        self.y: scores matrix torch.tensor (6497,1)
        """
        wine_quality = fetch_ucirepo(id=186) 
        # data (as pandas dataframes) 
        X = wine_quality.data.features 
        y = wine_quality.data.targets 
        # X and y from dataframe to tensor
        self.X = torch.tensor(X.values).float()
        self.y = torch.tensor(y.values).float()
        print(f"Loaded full wine quality dataset with shape: {self.X.shape},{self.y.shape}")
        
    def load_split_data(self, portion) -> None:
        """
        load a (total-size*portion) slice of the dataset; random split
        """
        if portion > 0.5 or portion <= 0:
            raise ValueError("Invalid! portion should be in [0,0.5] for algorithm running safety")
        # split data
        split_size = int(portion * self.X.shape[0])
        permute = torch.randperm(self.X.shape[0])
        indices_train = permute[:split_size]
        indices_test = permute[split_size:split_size + int(0.5 * split_size)]
        indices_calib = permute[split_size:split_size + int(0.5 * split_size)]

        self.X_train = self.X[indices_train]
        self.y_train = self.y[indices_train]
        self.X_test = self.X[indices_test]
        self.y_test = self.y[indices_test]
        self.X_calib = self.X[indices_calib]
        self.y_calib = self.y[indices_calib]
        print(f"Loaded split wine quality dataset \n 
               with train shape: {self.X_train.shape} \n 
               test size{self.X_test.shape}\n \
               calibration size {self.X_calib.shape} ")

    def get_data(self) -> tuple:
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def get_calibration_set(self) -> tuple:
        return self.X_calib, self.y_calib

    
    
