import torch
from utils import neg_log

def get_bce_loss(labels, preds, masks=None):
    
    batch_size = int(labels.size(0))
    num_classes = int(labels.size(1))
    demon_mtx = (batch_size * num_classes) * torch.ones_like(preds)
    
    loss_mtx = torch.zeros_like(preds)
    loss_mtx[labels == 1] = neg_log(preds[labels == 1])
    loss_mtx[labels == 0] = neg_log(1 - preds[labels == 0])
    
    if isinstance(masks, torch.Tensor) :
        mask_val = masks.any(dim=1).float().unsqueeze(1)    # if any true
        loss_mtx = loss_mtx * mask_val
    
    loss = (loss_mtx / demon_mtx).sum()

    return loss

def get_cap_loss(pseudo_labels, alpha=None, beta=None, masks=None):
    if alpha == None or beta == None:
        return 0
    
    batch_size = int(pseudo_labels.size(0))
    num_classes = int(pseudo_labels.size(1))
    demon_mtx = (batch_size * num_classes) * torch.ones_like(pseudo_labels)
    
    loss_mtx = pseudo_labels.clone()
    loss_mtx = alpha * loss_mtx + beta * (1-loss_mtx) 
    
    if isinstance(masks, torch.Tensor) :
        mask_val = masks.any(dim=1).float().unsqueeze(1)    # if any true
        loss_mtx = loss_mtx * mask_val
    
    loss = (loss_mtx / demon_mtx).sum()

    return loss

def compute_batch_loss(output_net, y_lb, unlabeled=False, lambda_u=None, alpha=None, beta=None):
    if unlabeled:
        assert lambda_u == None
        logits_lb, pseudo_labels, s_logits_ulb, masks = output_net
        lb_loss = get_bce_loss(y_lb, logits_lb)
        cap_loss = get_cap_loss(pseudo_labels, alpha, beta, masks)
        ulb_loss = get_bce_loss(pseudo_labels, s_logits_ulb, masks) - cap_loss
        return lb_loss + lambda_u * ulb_loss 
    else:
        logits_lb = output_net
        return get_bce_loss(y_lb, logits_lb)