import torch,sys,os,time
from torch import nn
from tqdm import tqdm
import random
import numpy as np
from argparse import ArgumentParser
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(main_folder_path)

class ConformalPredictionMethods:
    def __init__(self):
        pass

    @staticmethod
    def weighted_quantile(x, q, w):
        """
        x: torch.tensor, shape (n)
        q: float, quantile value
        """
        if not isinstance(x, torch.Tensor):
            raise ValueError("x must be a torch.tensor")
        if w.shape[0] != x.shape[0]:
            raise ValueError("weights must have the same length as x")
        if not (0 < q < 1):
            raise ValueError("quantile value must be in (0,1)")
        w = w/torch.sum(w)
        x ,idx = torch.sort(x)
        emp_cdf = torch.cumsum(w[idx],dim=0)
        idx = torch.nonzero(emp_cdf >= q, as_tuple=False)[0][0]
        return x[idx]
    
if __name__ == '__main__':
    method = ConformalPredictionMethods()
    x = torch.tensor([1,2,3,4,5,6,7,8,9,10],dtype=torch.float32)
    w00 = torch.ones_like(x)
    q = 0.1
    print(method.weighted_quantile(x,q,w00))
        