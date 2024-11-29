import torch
import torchvision
import torchvision.transforms as transforms
import os
import sys
from baseclasses.baseclass import BaseDataLoader
from torch.utils.data import DataLoader, RandomSampler

class MNISTLoader(BaseDataLoader):
    def __init__(self, args):
        super(MNISTLoader, self).__init__(args)
    
    def load_full_data(self):
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
    
    def load_split_data(self, portion):
        # use *portion of the train set and test set of MNIST
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
        
    
