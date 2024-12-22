import numpy as np
import pandas as pd
import random
from algorithms.conformal import Conformal
import os

random.seed(123)

if __name__=="__main__":
    # dataset QFT
    file_path = './datasets/QSAR Fish Toxicity.csv'
    data = pd.read_csv(file_path, header=0, sep=";")

    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()
    N = X.shape[0] # number of observations
    T = 100
    cov = np.zeros(T)
    length = np.zeros(T)

    # do random split into two equal sets
    for i in range(0, T):
        if (i + 1) % 20 == 0:
            print(i + 1, "..", sep="")
        idx = np.random.choice(range(0, N), size=int(N * 0.5), replace=False)
        noidx = np.setdiff1d(range(0, N), idx)

        X0 = X[idx, :] # training set
        y0 = y[idx]
        X00 = X[noidx, :] # calibration set 
        y00 = y[noidx]
        n00 = X00.shape[0] # size of calibration set
        Split_CP_QFT = Conformal(X0=X0, y0=y0, X00=X00, y00=y00, rho=0.5, alpha=0.1, weight=None, train_model="lasso")
        cov[i] = np.mean(Split_CP_QFT.split_weight_conformal()[0])
        length[i] = np.median(Split_CP_QFT.split_weight_conformal()[1])

    print("Coverage=", np.mean(cov), sep="")
    print("Average Length=", np.mean(length), sep="")
