import copy
import torch
import torch.nn as nn

class CapNet(nn.Module):
    def __init__(self, network, num_classes, bs_counter=0, n_0=1, n_1=1, T=1, semi_mode=False, device='cuda'):
        super(CapNet, self).__init__()
        self.network = copy.deepcopy(network)
        self.num_classes = num_classes
        self.T = T
        self.n_0 = n_0
        self.n_1 = n_1
        self.bs_counter = bs_counter
        self.semi_mode = semi_mode
        self.device = device
        self.true_bank = torch.zeros(self.num_classes).to(self.device)
    
    def forward(self, X_lb, y_lb, w_ulb=None, s_ulb=None):
        # X_lb (b, c, h, w), y_lb(b, prob)
        # change to X_lb(b, h, w, c)
        X_lb = X_lb.to(self.device)
        y_lb = y_lb.to(self.device)
        num_lb = X_lb.shape[0]
        
        self.true_bank += torch.sum((y_lb == 1), dim=0).to(self.device)
        self.bs_counter += num_lb

        # semi_mode == True => train with both labeled and unlabeled data
        if self.semi_mode:
            
            # X_ulb = (w_ulb, s_ulb)
            num_ulb = w_ulb.shape[0]
            inputs = torch.cat((X_lb, w_ulb, s_ulb), dim=0)
            logits = self.network(inputs)
            lb_logits = torch.sigmoid(logits[:num_lb] / self.T)
            w_logits, s_logits = logits[num_lb:].chunk(2)
            
            # choose indexes from sorted soft_labels (f(x)) 
            # so that only f(x) (%) is >= t_a
            soft_labels = torch.sigmoid(w_logits / self.T)
            
            # first transpose the sft_tp => (num_classes, sorted(batch))
            sft_tp = soft_labels.clone().permute(1, 0)

            # gamma hold distribution of positve label in labeled dataset
            # and ro hold distribution of negative label in labeled dataset
            gamma = self.true_bank / self.bs_counter 
            ro = 1 - gamma
            gamma = self.n_0 * gamma
            ro = self.n_1 * ro

            # calculate t_a (high threshold) and t_b (low threshold)
            sft_, _ = sft_tp.sort(descending=True)

            # init t_a and t_b
            t_a = torch.zeros(self.num_classes).to(self.device)
            t_b = torch.zeros(self.num_classes).to(self.device)

            # ids of high and low threshold (ID is from 0 to batch-1)
            t_a_ids = ((num_ulb-1) * gamma).int()
            t_b_ids = ((num_ulb-1) * ro).int()

            # calculate t_a and t_b base on t_a_ids, t_b_ids
            for i in range(self.num_classes):
                t_a = sft_[i, t_a_ids[i]]
                t_b = sft_[i, t_b_ids[i]]

            # make pseudo_labels and masks
            masks = torch.where((soft_labels <= t_b) | (soft_labels >= t_a), 1, 0).float()
            pseudo_labels = torch.where(soft_labels >= t_a, 1, -1).long()                   
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