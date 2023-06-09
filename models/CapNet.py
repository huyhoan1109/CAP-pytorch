import copy
import torch
import torch.nn as nn

class CapNet(nn.Module):
    def __init__(self, network, num_classes, device, bs_counter=0, n_0=0.95, n_1=0.95, T=1, semi_mode=False):
        super(CapNet, self).__init__()
        self.num_classes = num_classes
        self.T = T
        self.n_0 = n_0
        self.n_1 = n_1
        self.semi_mode = semi_mode
        self.device = device
        self.network = network.to(self.device)
        self.t_a_init = torch.zeros(self.num_classes).to(self.device)
        self.t_b_init = torch.zeros(self.num_classes).to(self.device)
    
    def forward(self, X_lb, y_lb, w_ulb=None, s_ulb=None, true_dist=None):
        # X_lb (b, c, h, w), y_lb(b, prob)
        # change to X_lb(b, h, w, c)
        X_lb = X_lb.to(self.device)
        y_lb = y_lb.to(self.device)
        num_lb = X_lb.shape[0]
        
        # semi_mode == True => train with both labeled and unlabeled data
        if self.semi_mode:
            
            # X_ulb = (w_ulb, s_ulb)
            num_ulb = w_ulb.shape[0]
            w_ulb = w_ulb.to(self.device)
            s_ulb = s_ulb.to(self.device)
            true_dist = true_dist.to(self.device)
            inputs = torch.cat((X_lb, w_ulb, s_ulb), dim=0)
            logits = self.network(inputs)
            lb_logits = torch.sigmoid(logits[:num_lb] / self.T)
            w_logits, s_logits = torch.sigmoid(logits[num_lb:]).chunk(2)
            
            # choose indexes from sorted soft_labels (f(x)) 
            # so that only f(x) (%) is >= t_a

            # first transpose the sft_tp => (num_classes, sorted(batch))
            sft_tp = w_logits.clone().permute(1, 0)

            # gamma hold distribution of positve label in labeled dataset
            # and ro hold distribution of negative label in labeled dataset
            gamma = true_dist 
            gamma = gamma
            ro = 1 - gamma
            gamma = self.n_0 * gamma
            ro = self.n_1 * ro

            # calculate t_a (high threshold) and t_b (low threshold)
            sft_, _ = sft_tp.sort(descending=True)

            # init t_a and t_b
            t_a = self.t_a_init
            t_b = self.t_b_init

            # ids of high and low threshold (ID is from 0 to batch-1)
            t_a_ids = ((num_ulb-1) * gamma).int()
            t_b_ids = ((num_ulb-1) * ro).int()

            # calculate t_a and t_b base on t_a_ids, t_b_ids
            for i in range(self.num_classes):
                t_a = sft_[i, t_a_ids[i]]
                t_b = sft_[i, t_b_ids[i]]

            # make pseudo_labels and masks
            masks = torch.where((w_logits <= t_b) | (w_logits >= t_a), 1, 0).float()
            pseudo_labels = torch.where(w_logits >= t_a, 1, 0).float()                
            results = {
                'logits': lb_logits,
                'pseudo_lb': pseudo_labels,
                's_logits': s_logits,
                'masks': masks,
                't_a': t_a,
                't_b': t_b
            } 
        else:
            # else train with labeled data and update gamma
            lb_logits = self.network(X_lb)
            lb_logits = torch.sigmoid(lb_logits / self.T)
            results = {
                'logits': lb_logits
            } 
        return results