import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import sys
from ucimlrepo import fetch_ucirepo 
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from baseclasses.baseclass import BaseDataLoader
from torch.utils.data import DataLoader, RandomSampler
from architectures.mlpRegressor import MLP_Regressor

class CSVsLoader(BaseDataLoader):
    def __init__(self, _args) -> None:
        """
        five datasets in UCI MACHINE LEARNING loaded from local files;
        WQ: Wine Quality
        CCS: Concrete Compressive Strength
        PT: Parkinsons Telemonitoring
        QFT: QSAR Fish Toxicity
        YH: Yacht Hydrodynamics
        """
        super(CSVsLoader, self).__init__(_args)
        self.args = _args
        self.device = self.args.device  
        self.DATASET_LIST = self.args.DATASET_LIST
    def load_full_data(self):
        """
        DATASETS_LIST = ['WQ','CCS','PT','QFT','YH']; WQ uses the WineQualityLoader, NOT THIS LOADER
        """
        if self.args.dataset not in self.args.DATASET_LIST:
            raise ValueError(f"Invalid dataset name: {self.args.dataset}")
        if self.args.dataset =='WQ':
            red_wine= pd.read_csv('datasets/winequality-red.csv', sep=';')
            white_wine = pd.read_csv('datasets/winequality-white.csv', sep=';')
            red_wine['type'] = 'red'
            white_wine['type'] = 'white'
            data = pd.concat([red_wine, white_wine], ignore_index=True)
        elif self.args.dataset =='CCS':
            file_path = './datasets/Concrete Compressive Strength.xls'
            data = pd.read_excel(file_path, header=0)
        elif self.args.dataset =='QFT':
            file_path = './datasets/QSAR Fish Toxicity.csv'
            data = pd.read_csv(file_path, header=0, sep=";")
        elif self.args.dataset =='PT':
            file_path = './datasets/Parkinsons Telemonitoring.data'
            data = pd.read_csv(file_path, header=0)
        elif self.args.dataset =='YH':
            file_path = './datasets/Yacht Hydrodynamics.data'
            data = pd.read_csv(file_path, header=0, sep='\s+')
        
        if self.args.dataset == "WQ":
            X = data.iloc[:, :-2].to_numpy()
            y= data.iloc[:, -2].to_numpy()
            N = X.shape[0]
            INPUT_DIM = X.shape[1]
            self.X = torch.tensor(X).float().to(self.device)
            self.y = torch.tensor(y).float().to(self.device)
            self.N = N
            self.INPUT_DIM = INPUT_DIM
        else:
            X = data.iloc[:, :-1].to_numpy()
            y = data.iloc[:, -1].to_numpy()
            N = X.shape[0]
            INPUT_DIM = X.shape[1]
            self.X = torch.tensor(X).float().to(self.device)
            self.y = torch.tensor(y).float().to(self.device)
            self.N = N
            self.INPUT_DIM = INPUT_DIM
        print(f"Loaded full dataset {self.args.dataset} with shape: {self.X.shape},{self.y.shape}")

    def load_split_data(self):
        idx = np.random.choice(range(0, self.N), size=int(self.N * 0.5), replace=False)
        noidx = np.setdiff1d(range(0, self.N), idx)

        self.X0 = self.X[idx, :] # training set
        self.y0 = self.y[idx]
        self.X00 = self.X[noidx, :] # calibration set 
        self.y00 = self.y[noidx]
        self.n00 = self.X00.shape[0] # size of calibration set
        print(f"Loaded split dataset {self.args.dataset} train shape: {self.X0.shape} test size{self.X00.shape} ")
    
    def make_MLP_Regressor(self):
        """
        Setup the simple MLP regressor model
        """
        self.model = MLP_Regressor(task_type='regression', input_dim=self.INPUT_DIM, 
                                   output_dim=1, dropout_prob=0.2, hidden_dims=[64, 64, 32])
        self.task_type = self.model.task_type
        if self.task_type == 'regression':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.epochs = self.args.epochs
        self.batchsize = 128
        self.feat_iter = 128 # M in the paper's algorithm 2, for calculating nonconformity scores
        self.model.to(self.device)
        print(f"MLP Regressor model setup with input dim: {self.INPUT_DIM}")

    def train_MLP_Regressor(self):
        self.train_X = self.X0
        self.train_y = self.y0.unsqueeze(-1)
        self.test_X = self.X00
        self.test_y = self.y00.unsqueeze(-1)

        train_loss_logger = []
        test_loss_logger = []
        for epoch in tqdm(range(self.epochs),desc="MLP Regression Model Training"):
            self.model.train()
            epoch_loss = 0.0
            for i in range(0,self.train_X.shape[0],self.batchsize):
                X_batch = self.train_X[i:min(i+self.batchsize,self.train_X.shape[0]),:]
                y_batch = self.train_y[i:min(i+self.batchsize,self.train_X.shape[0]),:]
                # optimizer run
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            train_loss_logger.append(epoch_loss)
            self.model.eval()
            with torch.no_grad():
                test_loss = 0.0
                for i in range(0, self.test_X.shape[0], self.batchsize):
                    X_test_batch = self.test_X[i:min(i + self.batchsize, self.test_X.shape[0]), :]
                    y_test_batch = self.test_y[i:min(i + self.batchsize, self.test_X.shape[0]), :]
                    test_outputs = self.model(X_test_batch)
                    test_loss += self.criterion(test_outputs, y_test_batch).item()
                test_loss_logger.append(test_loss)
            if epoch % 250 == 0:
                print(f"[{epoch + 1}] current epoch total loss: {epoch_loss}")  
        print(f"Training finished, final loss: {train_loss_logger[-1]}")
        self.train_loss_logger = train_loss_logger
        self.test_loss_logger = test_loss_logger
    

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
        print(f"Loaded split wine quality dataset with train shape: {self.X_train.shape} \n test size{self.X_test.shape},calibration size {self.X_calib.shape} ")

    def get_data(self) -> tuple:
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def get_calibration_set(self) -> tuple:
        return self.X_calib, self.y_calib



    
    
