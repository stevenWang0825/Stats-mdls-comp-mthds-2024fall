import sys
import os
import time
from torch import nn

class BaseTrainer:
    def __init__(self, args):
        self.args = args
        self.model = None
        
    def train_setup(self, train_data, valid_data = None):
        raise NotImplementedError
    
    def train(self):
        raise NotImplementedError
    
    def test_setup(self,test_data):
        raise NotImplementedError
    
    def test(self, test_data):
        raise NotImplementedError
    
class BaseDataLoader:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        
    def load_data(self):
        raise NotImplementedError

    def get_data(self):
        raise NotImplementedError