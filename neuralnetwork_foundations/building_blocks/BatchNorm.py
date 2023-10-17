import torch
import torch.nn as nn

class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Initialize learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Initialize running mean and variance
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
    
    
    def forward(self, x):
        if self.training:
            # Calculate mean and variance for each mini-batch
            mean = torch.mean(x, dim=0)
            var = torch.var(x, dim=0, unbiased=False)
            
            # Update running mean and variance
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
            # Normalize the input
            x_normalized = (x - mean) / torch.sqrt(var + self.eps)
            
        else:
            # Use running mean and variance during inference
            x_normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        
        # Scale and shift the normalized input
        output = self.gamma * x_normalized + self.beta
        return output

