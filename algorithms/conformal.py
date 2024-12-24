import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
import random

random.seed(123)

class Conformal:
    def __init__(self,
                 X0,
                 y0,
                 X00,
                 y00,
                 rho,
                 alpha,
                 weight,
                 train_model):
        """
        :param X0: training set 
        :param X00: calibration set
        :param weight: CP weights
        """
        self.X0 = X0
        self.y0 = y0
        self.X00 = X00
        self.y00 = y00
        self.rho = rho
        self.alpha = alpha
        self.weight = weight
        self.train_model = train_model

    def weighted_quantile(self, x, q, w):
        w = w / np.sum(w)
        emp_cdf = np.cumsum(w)
        idx = np.where(emp_cdf >= q)[0][0]
        return x[idx]

    def split_weight_conformal(self):
        rho = self.rho
        X0 = self.X0
        X00 = self.X00
        y0 = self.y0
        y00 = self.y00
        alpha = self.alpha
        weight = self.weight
        train_model = self.train_model
        n = X0.shape[0]
        n00 = X00.shape[0]
        if weight is None:
            weight = np.ones(n + n00)

        lo = np.zeros(n00)
        up = np.zeros(n00)
        cov = np.zeros(n00)
        length = np.zeros(n00)

        i1 = np.random.choice(range(0, n), size=int(n * rho), replace=False)
        i2 = np.setdiff1d(range(0, n), i1)
        n1 = len(i1)
        n2 = len(i2)

        if train_model == "linear":
            model = LinearRegression()
            model.fit(X0, y0)
            predict_model = model.predict
        elif train_model == "randomforest":
            model = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=123)
            model.fit(X0, y0)
            predict_model = model.predict
        elif train_model == "lasso":
            model = Lasso(alpha=1.0)
            model.fit(X0, y0)
            predict_model = model.predict
        elif train_model == "ridge":
            model = Ridge(alpha=1.0)
            model.fit(X0, y0)
            predict_model = model.predict

        cal_scores = np.abs(y0[i2] - predict_model(X0[i2, :]))
        ordering = np.argsort(cal_scores)
        cal_scores = cal_scores[ordering]
        for i in range(0, n00):
            q = self.weighted_quantile(np.append(cal_scores, float('inf')), 1-alpha, np.append(weight[i2][ordering], weight[n+i]))
            up[i] = predict_model(X00[i, :].reshape(1, -1)).item() + q
            lo[i] = predict_model(X00[i, :].reshape(1, -1)).item() - q
        length = up - lo
        cov = np.where((y00 >= lo) & (y00 <= up), 1, 0)
        # print(np.shape(cov))
        # print(np.shape(up))
        # print(np.shape(length))
        return cov, length


if __name__=="__main__":
    file_path = './datasets/Airfoil Self-Noise.txt'
    data = np.loadtxt(file_path)
    print(data.shape)

    X = data[:, :-1] # features
    # do some transformation on some features
    X[:, 0] = np.log(X[:, 0]) 
    X[:, 4] = np.log(X[:, 4])
    y = data[:, -1] # target or response
    N = X.shape[0] # number of observations
    # do random split into two equal sets
    idx = np.random.choice(range(0, N), size=int(N * 0.5), replace=False)
    noidx = np.setdiff1d(range(0, N), idx)
    
    X0 = X[idx, :] # training set (pre-training and calibration)
    y0 = y[idx]
    X00 = X[noidx, :] # test set 
    y00 = y[noidx]
    n00 = X00.shape[0] # size of test set

    conformal = Conformal(X0=X0, y0=y0, X00=X00, y00=y00, rho=0.5, alpha=0.1, weight=None, train_model="lasso")
    print(np.sum(conformal.split_weight_conformal()[0]) / n00)

