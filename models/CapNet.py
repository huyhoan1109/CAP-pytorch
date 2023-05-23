import copy
import torch
import torch.nn as nn

class CapNet(nn.Module):
    def __init__(self, network, num_classes, n_0=1, n_1=1, run_time=0, T=1, semi_mode=False):
        super(CapNet, self).__init__()
        self.network = copy.deepcopy(network)
        self.num_classes = num_classes
        self.T = T
        self.n_0 = n_0
        self.n_1 = n_1
        self.run_times = run_time
        self.semi_mode = semi_mode
        self.true_bank = torch.zeros(self.num_classes)
    
    def forward(self, X_lb, y_lb, X_ulb=None):
        # X_lb (b, c, h, w), y_lb(b, prob)
        # change to X_lb(b, h, w, c)
        X = X_lb.permute(0, 3, 1, 2)
        num_lb = X_lb.shape[0]
        
        if self.semi_mode:
            
            # X_ulb = (w_ulb, s_ulb)
            num_ulb = X_ulb[0].shape[0]
            w_ulb = X_ulb[0].permute(0, 3, 1, 2)
            s_ulb = X_ulb[1].permute(0, 3, 1, 2)
            inputs = torch.cat((X, w_ulb, s_ulb), dim=0)
            logits = self.network(inputs)
            lb_logits = torch.sigmoid(logits[:num_lb] / self.T)
            w_logits, s_logits = logits[num_lb:].chunk(2)
            
            # choose indexes from sorted soft_labels (f(x)) 
            # so that only f(x) (%) is >= t_a
            soft_labels = torch.sigmoid(w_logits / self.T)
            sft_tp = soft_labels.clone().permute(1, 0)
            thresh_indices = (num_ulb * self.gamma).int()
            sft_mx, _ = sft_tp.sort(descending=True) 
            t_a = torch.zeros(self.num_classes)      
            for i in range(self.num_classes):
                t_a[i] = sft_mx[i, thresh_indices[i]]
            t_b = 1 - t_a
            t_a = self.n_0 * t_a
            t_b = self.n_1 * t_b
            
            # make pseudo_labels and masks
            masks = torch.where((soft_labels <= t_b) | (soft_labels >= t_a), 1, 0).float()
            pseudo_labels = torch.where(soft_labels >= 0.5, 1, -1).long()       
            
            results = {
                'lb_logits': lb_logits,
                'pseudo_lb': pseudo_labels,
                's_logits': s_logits,
                'masks': masks,
                't_a': t_a,
                't_b': t_b
            } 
        
        else:
            
            # else train with labeled data and update gamma
            lb_logits = self.network(X)
            self.run_times += 1
            self.true_bank += torch.sum((y_lb == 1), dim=0).clone()
            self.gamma = self.true_bank / (num_lb * self.run_times) 
            lb_logits = torch.sigmoid(lb_logits / self.T)
            
            results = {
                'lb_logits': lb_logits
            } 
        return results