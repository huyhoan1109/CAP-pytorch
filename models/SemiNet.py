import copy
import torch.nn as nn

class SemiNet(nn.Module):
    def __init__(self, network, semi_mode=False) -> None:
        super(SemiNet, self).__init__()
        self.network = copy.deepcopy(network)
        self.semi_mode = semi_mode
    
    def forward(self, X_lb, y_lb, X_ulb=None):
        pass