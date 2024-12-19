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

from baseclasses.baseclass import BaseTrainer
from architectures.mlpRegressor import MLP_Regressor
from dataloaders.dataloaders import WineQualityLoader
from algorithms.feature_CP import ConformalPredictionMethods


class MLPRegressorTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.model = MLP_Regressor(task_type='regression', input_dim=11, output_dim=1, dropout_prob=0.2, hidden_dims=[64, 64, 32])
        self.task_type = self.model.task_type
        if self.task_type == 'regression':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.epochs = 1000
        self.batchsize = 128
        self.feat_iter = 128 # M in the paper's algorithm 2, for calculating nonconformity scores
        self.device = args.device
        self.model.to(self.device)

    def train_setup(self, train_data, valid_data=None):
        self.train_data = train_data.to(self.device)
        self.train_X = self.train_data[:,:-1]
        self.train_y = self.train_data[:,-1].reshape(-1,1)
    
    def train(self):
        """
        require:
        train_data [torch.tensor]: (n_samples, n_features+1)
        train_X [torch.tensor]: (n_samples, n_features)
        train_y [torch.tensor]: (n_samples, 1)
        """
        self.model.train()
        loss_logger = []
        for epoch in tqdm(range(self.epochs),desc="MLP Regression Model Training"):
            epoch_loss = 0.0
            for i in range(0,self.train_data.shape[0],self.batchsize):
                X_batch = self.train_X[i:min(i+self.batchsize,self.train_data.shape[0]),:]
                y_batch = self.train_y[i:min(i+self.batchsize,self.train_data.shape[0]),:]
                # optimizer run
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss_logger.append(loss.item())
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            if epoch % 50 == 0:
                print(f"[{epoch + 1}] current epoch total loss: {epoch_loss}")
    
    def test_setup(self, test_data):
        return super().test_setup(test_data)
    
    def test(self):
        return super().test()
    
    def calibration_setup(self, calib_data):
        self.calib_data = calib_data.to(self.device)
        print(f"Setup calibration set, data shape: {self.calib_data.shape}")

    # def _conformity_score(self, calib_data_point):
    #     """
    #     the unvectorized version of calculating conformity scores.
    #     calib_data_point: [number_of_features+1]: feature vector + Y_truth(true Y value)
    #     u_feat: output of the feature function 'f'
    #     Y_truth: the true Y value
    #     u_surrogate: surrogate feature u
    #     """
    #     self.model.eval()
    #     u_temp = self.model.feature_func_forward(calib_data_point[:,:-1].unsqueeze(0))
    #     u_feat = u_temp.clone().detach().requires_grad_(True) # this is making u_feat a leaf node
    #     Y_truth = calib_data_point[:,-1]
    #     optimizer = torch.optim.SGD([u_feat], lr=1e-2,momentum=0.9)
    #     for _ in range(self.feat_iter):
    #         loss = (self.model.predictor_head_forward(u_feat) - Y_truth)**2
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     u_surrogate = u_feat
    #     return torch.norm(u_temp - u_surrogate,p=2).item()
    
    # def conformity_scores(self):
    #     """
    #     calculate conformity scores for all calibration data points
    #     calib_data: [calib_size, number_of_features+1]: each row = feature vectors + Y_truth
    #     """
    #     self.model.eval()
    #     start_time = time.time() 
    #     conformity_scores = []
    #     for i in tqdm(range(self.calib_data.shape[0]),desc="calculating scores on calib set"):
    #         conformity_scores.append(self._conformity_score(self.calib_data[i]))
    #     end_time = time.time()
    #     print(f"conformity scores used time: {end_time-start_time}s")
    #     return np.array(conformity_scores)
    """
    WARNING:
    THE METHODS ABOVE ARE ABORTED BECAUSE OF ITS DEPENDENCE ON OUTER PACKAGES.
    THESE PACKAGES DATE BACK TO 2020 SO THEY MAY BE INCOMPATIBLE WITH THE CURRENT ENVIRONMENT.
    IT ALSO REQUIRES MILP SOLVING, BUT THE SOLVER MAY BE UNAVAILABLE FOR US.
    THIS DOES NOT MEAN THEY ARE INCORRECT.
    """
    
    def fast_conformity_scores(self):
        """
        the score calculation method of "FFCP"
        self.calib_data:[calib_size, number_of_features+1]
        """
        self.model.eval()
        start_time = time.time()
        u_feat = self.model.feature_func_forward(self.calib_data[:,:-1])
        y_pred = self.model.predictor_head_forward(u_feat)
        abs = torch.abs(y_pred.squeeze(-1) - self.calib_data[:,-1])
        grad_norm = torch.norm(torch.autograd.grad(y_pred[0],u_feat,allow_unused=True,retain_graph=True)[0][0],p=2)
        scores = abs/grad_norm
        end_time = time.time()
        print(f"fast conformity scores used time: {end_time-start_time}s")
        return scores,grad_norm
        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='device to run the model on')
    args = parser.parse_args()
    print(f"using device:{args.device}")

    dataloader_instance = WineQualityLoader(args)
    dataloader_instance.load_full_data()
    dataloader_instance.load_split_data(0.5)
    X_train,y_train,x_test,y_test = dataloader_instance.get_data()
    X_calib,y_calib = dataloader_instance.get_calibration_set()

    trainer_instance = MLPRegressorTrainer(args)
    # trainer_instance.train_setup(train_data=torch.cat((X_train,y_train),dim=1), valid_data=None)
    # trainer_instance.train()
    
    trainer_instance.model.eval()
    trainer_instance.calibration_setup(calib_data=torch.cat((X_calib,y_calib),dim=1))
    scores, grad_norm = trainer_instance.fast_conformity_scores()

    cpmethods = ConformalPredictionMethods()
    Q0_1 = cpmethods.weighted_quantile(scores,0.9,torch.ones_like(scores))
    print(f"Q(0.1): {Q0_1}")

    
    
    
    
