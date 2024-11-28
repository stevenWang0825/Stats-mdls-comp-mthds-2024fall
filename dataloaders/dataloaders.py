import torch
import torchvision
import torchvision.transforms as transforms
import os
from baseclasses.baseclass import BaseDataLoader

class MNISTLoader(BaseDataLoader):
    def __init__(self, args):
        super(MNISTLoader, self).__init__(args)
    
    def load_data(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        # check if data is already downloaded
        if not os.path.exists('./datasets/MNIST'):
            print("Downloading MNIST dataset...")
            download = True
        else:
            print("MNIST dataset already downloaded.")
            download = False

        # load data
        self.trainset = torchvision.datasets.MNIST(root='./datasets', train=True, download=download, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=64, shuffle=True)
        self.testset = torchvision.datasets.MNIST(root='./datasets', train=False, download=download, transform=transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=64, shuffle=False)
    
    def get_data(self):
        return self.trainloader, self.testloader
    
