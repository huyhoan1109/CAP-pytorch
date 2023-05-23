import copy
import torch
import torch.nn as nn

class CapNet(nn.Module):
    def __init__(self, network, num_classes, T=1, semi_mode=False):
        super(CapNet, self).__init__()
        self.network = copy.deepcopy(network)
        self.num_classes = num_classes
        self.T = T
        self.run_times = 0
        self.semi_mode = semi_mode
        self.init_model()

    def init_model(self):
        self.bank = torch.zeros(self.num_classes)
        self.low_thresh = torch.zeros(self.num_classes)
    
    def forward(self, X_lb, y_lb, X_ulb=None):
        # X_lb (b, h, w, c), y_lb(b, prob)
        X = X_lb.permute(0, 3, 1, 2).clone()
        num_lb = X_lb.shape[0]
        if self.semi_mode:
            w_ulb = X_ulb[0].permute(0, 3, 1, 2).clone()
            s_ulb = X_ulb[1].permute(0, 3, 1, 2).clone()
            inputs = torch.cat((X, w_ulb, s_ulb))
            logits = self.network(inputs)
            lb_logits = torch.sigmoid(logits[:num_lb] / self.T)
            w_logits, s_logits = logits[num_lb:].chunk(2)
            soft_labels = torch.sigmoid(w_logits / self.T)
            masks = torch.where((soft_labels <= (self.low_thresh)) | (soft_labels >= (1-self.low_thresh)), 1, 0).float()
            pseudo_labels = torch.where(soft_labels >= 0.5, 1, -1).long() 
            results = {
                'lb_logits': lb_logits,
                'pseudo_lb': pseudo_labels,
                's_logits': s_logits,
                'masks': masks,
                'low_thresh': self.low_thresh
            } 
            return results
        else:
            lb_logits = self.network(X)
            self.run_times += 1
            self.bank += torch.sum((y_lb == 1), dim=0).clone()
            self.low_thresh = self.bank / (num_lb * self.run_times) 
            lb_logits = torch.sigmoid(lb_logits / self.T)
            results = {
                'lb_logits': lb_logits,
            } 
            return results