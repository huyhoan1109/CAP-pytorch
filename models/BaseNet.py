import copy
import torch
import torch.nn as nn
from MA.EMA import EMA

class BaseNet(nn.Module):
    def __init__(
        self,
        network, 
        T=1,
        ema_decay=False,
        ):
        super(BaseNet).__init__()
        self.network = copy.deepcopy(network)
        self.T = T
        self.ema_decay = ema_decay
        self.init_model()

    def init_model(self):
        if self.ema_decay:
            self.network = EMA(self.network, beta=self.ema_decay)

    def forward(self, X_lb, X_ulb=None, threshold_low=None, threshold_high=None):
        # X_lb (b, h, w, c), y_lb(b, prob)
        num_lb = X_lb.shape[0]
        if X_ulb == None:
            logits_lb = self.network(X_lb)
            return logits_lb
        else:
            # X_ulb = (w_ulb, s_ulb)
            w_ulb, s_ulb = X_ulb[0], X_ulb[1]
            inputs = torch.cat((X_lb, w_ulb, s_ulb))
            logits = self.network(inputs)
            logits_lb = logits[:num_lb]
            w_logits_ulb, s_logits_ulb = logits[num_lb:].chunk(2)
            soft_labels = torch.sigmoid(w_logits_ulb / self.T)
            masks = torch.where((soft_labels <= threshold_low) | (soft_labels >= threshold_high), 1, 0).float()
            pseudo_labels = torch.where(soft_labels >= 0.5, 1, -1).long()
            return logits_lb, pseudo_labels, s_logits_ulb, masks