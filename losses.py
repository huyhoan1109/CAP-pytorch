import torch
from utils import neg_log, AverageMeter
from models.CapNet import CapNet

def loss_asl(labels, preds, masks=None, gamma_neg=1, gamma_pos=1):

    batch_size = int(labels.size(0))
    num_classes = int(labels.size(1))
    demon_mtx = (batch_size * num_classes) * torch.ones_like(preds)
    
    preds_pos = preds[labels == 1]
    preds_neg = preds[labels == 0]

    loss_mtx = torch.zeros_like(preds)
    loss_mtx[labels == 1] = ((1 - preds_pos)**gamma_pos) * neg_log(preds_pos)
    loss_mtx[labels == 0] = (preds_neg**gamma_neg) * neg_log(1 - preds_neg)
    
    if isinstance(masks, torch.Tensor) :
        mask_val = masks.any(dim=1).float().unsqueeze(1)    # if any true
        loss_mtx = loss_mtx * mask_val
    
    loss = (loss_mtx / demon_mtx).sum()

    return loss

def loss_bce(labels, preds, masks=None):
    
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

def loss_cap(pseudo_labels, t_a=None, t_b=None, masks=None):
    if t_a == None or t_b == None:
        return 0

    batch_size = int(pseudo_labels.size(0))
    num_classes = int(pseudo_labels.size(1))
    demon_mtx = (batch_size * num_classes) * torch.ones_like(pseudo_labels)
    
    loss_mtx = pseudo_labels.clone()
    alpha = neg_log(t_a)
    beta = neg_log(t_b)
    loss_mtx = alpha * loss_mtx + beta * (1-loss_mtx) 
    
    if isinstance(masks, torch.Tensor) :
        mask_val = masks.any(dim=1).float().unsqueeze(1)    # if any true
        loss_mtx = loss_mtx * mask_val
    
    loss = (loss_mtx / demon_mtx).sum()

    return loss

def compute_batch_loss(args, logger, trackers, performances, batch, model, lambda_u, mode='train'):
    
    accuracy = None
    loss = 0
    
    if mode == 'train':
        X_lb = batch['lb']['X']
        y_lb = batch['lb']['y']
        X_ulb = batch['ulb']['X']
        preds = model(X_lb, y_lb, X_ulb)
        lb_logits = preds['logits']
        if args.use_asl:
            lb_loss = loss_asl(y_lb, lb_logits)
        else:
            lb_loss = loss_bce(y_lb, lb_logits)
        trackers['train']['lb_loss'].update(lb_loss.items())
        logger.log({'train/lb_loss': '{:05.3f}'.format(trackers['train']['lb_loss'].show())}, commit=False)
        loss = lb_loss
        if model.semi_mode:
            pseudo_lb = preds['pseudo_lb']
            s_logits = preds['s_logits']
            masks = preds['masks']
            if args.use_asl:
                ulb_loss = loss_asl(pseudo_lb, s_logits, masks)
            else:
                ulb_loss = loss_bce(pseudo_lb, s_logits, masks)
            trackers['train']['ulb_loss'].update(ulb_loss.items())
            logger.log({'train/ulb_loss': '{:05.3f}'.format(trackers['train']['ulb_loss'].show())}, commit=False)
            loss += lambda_u * ulb_loss
            
            if isinstance(model, CapNet):
                t_a = preds['t_a']
                t_b = preds['t_b']
                cap_loss = loss_cap(pseudo_lb, t_a, t_b, masks)
                trackers['train']['cap_loss'].update(cap_loss)
                logger.log({'train/ulb_loss': '{:05.3f}'.format(trackers['train']['cap_loss'].show())}, commit=False)
                loss -= lambda_u * cap_loss
        trackers['train']['main_loss'].update(loss)
        logger.log({'train/main_loss': '{:05.3f}'.format(trackers['train']['main_loss'].show())}, commit = False)
    
    elif mode == 'valid':
        X_valid = batch['valid']['X']
        y_valid = batch['valid']['y']
        
        prev = {}

        prev['semi_mode'] = model.semi_mode
        model.semi_mode = False
        if isinstance(model, CapNet):
            prev['update_bank'] = model.update_bank
            model.update_bank = False
        
        preds = model(X_valid, y_valid)['logits']
        loss = loss_asl(y_valid, preds)
        
        model.semi_mode = prev['semi_mode']
        if isinstance(model, CapNet):
            model.update_bank = prev['update_bank']
        
        trackers['valid']['loss'].update(loss)
        accuracy = performances['MAP'].scoring(y_valid, preds)
        logger.log({'val/loss': '{:05.3f}'.format(trackers['valid']['loss'].show())}, commit=False),
        logger.log({'val/acc: {:05.5f}'.format(accuracy)}, commit=False)
    
    return loss, accuracy