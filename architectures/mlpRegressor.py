import torch.nn.functional as F
import torch.nn as nn
import random
import torch
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class MLP_Regressor(nn.Module):
    def __init__(self, task_type, input_dim, output_dim, dropout_prob, hidden_dims) -> None:
        super(MLP_Regressor, self).__init__()
        if not isinstance(hidden_dims, list):
            raise ValueError('hidden dimensions must be a list')
        if not all(isinstance(x, int) for x in hidden_dims):
            raise ValueError('hidden dimensions must be integers')
        if not (0 <= dropout_prob <= 1):
            raise ValueError('dropout probability must be between 0 and 1')
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        dims = [input_dim] + hidden_dims
        self.dims = dims
        print(f"setup a MLP regression network structure with {self.dims}, dropout prob: {dropout_prob}")

        for i in range(len(dims) - 1):
            self.hidden_layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.batch_norms.append(nn.BatchNorm1d(dims[i + 1]))
            self.dropouts.append(nn.Dropout(p=dropout_prob))
        
        self.output_layer = nn.Linear(dims[-1], output_dim)
        self.task_type = task_type 
        if self.task_type not in ['regression', 'classification']:
            raise ValueError('task type must be either regression or classification')
    
    def forward(self, x) -> torch.Tensor:
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i](x)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropouts[i](x)
        x = self.output_layer(x)
        return x
    
    def feature_func_forward(self,x) -> torch.Tensor:
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i](x)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropouts[i](x)
        assert x.shape[1] == self.dims[-1]
        print(f"feature function output shape: {x.shape}")
        return x

# if __name__ == '__main__':