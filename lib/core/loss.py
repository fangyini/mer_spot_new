import torch
import torch.nn as nn
import torch.nn.functional as F
from focal_loss.focal_loss import FocalLoss


dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def abs_smooth(x):
    '''
    Sommth L1 loss
    Defined as:
        x^2 / 2        if abs(x) < 1
        abs(x) - 0.5   if abs(x) >= 1
    '''
    absx = torch.abs(x)
    minx = torch.min(absx, torch.tensor(1.0).type_as(dtype))
    loss = 0.5 * ((absx - 1) * minx + absx)
    loss = loss.mean()
    return loss


def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes, device=labels.device)  
    return y[labels]           


class Focal_loss(nn.Module): # [1/89.31, 1/1.5, 1/4.06, 1/1.95, 1/3.17]
    def __init__(self, alpha=[1/1.5, 1/4.06, 1/1.95, 1/3.17], gamma=0.5, num_classes=5, eps=1e-6):
        super(Focal_loss, self).__init__()
        #self.alpha = alpha
        self.alpha = torch.FloatTensor(alpha).type_as(dtype)
        self.gamma = gamma
        self.num_classes = num_classes
        self.eps = eps
        self.m = torch.nn.Softmax(dim=-1)

    def forward(self, x, y):
        #t = one_hot_embedding(y, 1 + self.num_classes)
        #t = t[:, 1:]
        t = one_hot_embedding(y, self.num_classes)

        p = x.sigmoid()
        pt = p * t + (1 - p) * (1 - t)  # pt = p if t > 0 else 1-p
        pt = pt.clamp(min=self.eps)  # avoid log(0)
        alpha = 0.25
        w = alpha * t + (1 - alpha) * (1 - t)  # w = alpha if t > 0 else 1-alpha
        loss = -(w * (1 - pt).pow(self.gamma) * torch.log(pt))
        res = loss.sum()
        return res
        
        with torch.no_grad():
            t = one_hot_embedding(y, self.num_classes)
            t = t[:, 1:]
            bs = t.size()[0]
            weight = self.alpha.unsqueeze(0).repeat(bs, 1)
        p = self.m(x)
        pt = p * t
        non_zero = torch.where(t > 0)
        pos = torch.where(y>0)
        neg = torch.where(y == 0)
        loss = -(weight[non_zero] * (1-pt[non_zero]).pow(self.gamma) * torch.log(pt[non_zero]))
        loss_pos = loss[pos] # 0.21
        loss_neg = loss[neg].sum() # 4.042
        res = loss.sum()
        return res # 4.2523, 16.8037, 4.2243, 16.1491


'''class Focal_loss(nn.Module):
    def __init__(self, class_weight=[1/89.31, 1/1.5, 1/4.06, 1/1.95, 1/3.17], gamma=2):
        super(Focal_loss, self).__init__()
        weights = torch.FloatTensor(class_weight).type_as(dtype)
        self.criterion = FocalLoss(gamma=gamma, weights=weights)
        self.m = torch.nn.Softmax(dim=-1)

    def forward(self, logits, target):
        prob = self.m(logits)
        loss = self.criterion(prob, target)
        return loss'''

'''class Focal_loss(nn.Module):
    def __init__(self, alpha=[1/89.31, 1/1.5, 1/4.06, 1/1.95, 1/3.17], gamma=2):
        super(Focal_loss, self).__init__()
        alpha = torch.FloatTensor(alpha).type_as(dtype)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss)
        index = torch.where(targets > 0)
        a = loss[index]
        loss = loss.mean()
        return loss'''


'''class Focal_loss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=[1/89.31, 1/1.5, 1/4.06, 1/1.95, 1/3.17], gamma=2, device='cpu'):
        weight = torch.FloatTensor(weight).type_as(dtype)
        super(Focal_loss, self).__init__(weight)
        # focusing hyper-parameter gamma
        self.gamma = gamma
        # class weights will act as the alpha parameter
        self.weight = weight
        # using deivce (cpu or gpu)
        self.device = DEVICE
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, _input, _target):
        focal_loss = 0

        for i in range(len(_input)):
            # -log(pt)
            cur_ce_loss = self.ce_loss(_input[i].view(-1, _input[i].size()[-1]), _target[i].view(-1))
            pt = torch.exp(-cur_ce_loss)
            cur_focal_loss = self.weight[_target[i]] * ((1 - pt) ** self.gamma) * cur_ce_loss
            focal_loss = focal_loss + cur_focal_loss

        focal_loss = focal_loss / self.weight.sum()
        return focal_loss.to(self.device)'''


def loss_function_ab(anchors_x, anchors_w, anchors_rx_ls, anchors_rw_ls, anchors_class,
                     match_x, match_w, match_scores, match_labels, cfg):
    '''
    calculate classification loss, localization loss and overlap_loss
    pmask, hmask and nmask are used to select training samples
    anchors_class: bs, sum_i(ti*n_box), nclass
    others: bs, sum_i(ti*n_box)
    '''
    target_rx = (match_x - anchors_x) / anchors_w
    target_rw = torch.log(match_w / anchors_w)

    match_scores = match_scores.view(-1)
    pmask = match_scores > cfg.TRAIN.FG_TH
    nmask = match_scores < cfg.TRAIN.BG_TH

    # classification loss
    keep = (pmask.float() + nmask.float()) > 0
    anchors_class = anchors_class.view(-1, cfg.DATASET.NUM_CLASSES)[keep]
    match_labels = match_labels.view(-1)[keep]
    #cls_loss_f = Focal_loss(num_classes=cfg.DATASET.NUM_CLASSES)
    cls_loss_f = Focal_loss()
    cls_loss = cls_loss_f(anchors_class, match_labels) / torch.sum(pmask)

    # localization loss
    if torch.sum(pmask) > 0:
        keep = pmask
        target_rx = target_rx.view(-1)[keep]
        target_rw = target_rw.view(-1)[keep]
        anchors_rx_ls = anchors_rx_ls.view(-1)[keep]
        anchors_rw_ls = anchors_rw_ls.view(-1)[keep]

        loc_loss = abs_smooth(target_rx - anchors_rx_ls) + abs_smooth(target_rw - anchors_rw_ls)
    else:
        loc_loss = torch.tensor(0.).type_as(cls_loss)
    # print('loss:', cls_loss.item(), loc_loss.item(), overlap_loss.item())

    return cls_loss, loc_loss


def sel_fore_reg(cls_label_view, target_regs, pred_regs):
    '''
    Args:
        cls_label_view: bs*sum_t
        target_regs: bs, sum_t, 1
        pred_regs: bs, sum_t, 1
    Returns:
    '''
    sel_mask = cls_label_view >= 1.0
    target_regs_view = target_regs.view(-1)
    target_regs_sel = target_regs_view[sel_mask]
    pred_regs_view = pred_regs.view(-1)
    pred_regs_sel = pred_regs_view[sel_mask]

    return target_regs_sel, pred_regs_sel


def iou_loss(pred, target):
    inter_min = torch.max(pred[:, 0], target[:, 0])
    inter_max = torch.min(pred[:, 1], target[:, 1])
    inter_len = (inter_max - inter_min).clamp(min=0)
    union_len = (pred[:, 1] - pred[:, 0]) + (target[:, 1] - target[:, 0]) - inter_len
    tious = inter_len / union_len
    loss = (1-tious).mean()
    return loss


def loss_function_af(cate_label, preds_cls, target_loc, pred_loc, cfg):
    '''
    preds_cls: bs, t1+t2+..., n_class
    pred_regs_batch: bs, t1+t2+..., 2
    '''
    batch_size = preds_cls.size(0)
    cate_label_view = cate_label.view(-1)
    cate_label_view = cate_label_view.type_as(dtypel)
    preds_cls_view = preds_cls.view(-1, cfg.DATASET.NUM_CLASSES)
    pmask = (cate_label_view > 0).type_as(dtype)

    if torch.sum(pmask) > 0:
        # regression loss
        mask = pmask == 1.0
        pred_loc = pred_loc.view(-1, 2)[mask]
        target_loc = target_loc.view(-1, 2)[mask]
        reg_loss = iou_loss(pred_loc, target_loc)
    else:
        reg_loss = torch.tensor(0.).type_as(dtype)
    # cls loss
    cate_loss_f = Focal_loss() #(num_classes=cfg.DATASET.NUM_CLASSES)
    cate_loss = cate_loss_f(preds_cls_view, cate_label_view) / (torch.sum(pmask) + batch_size)  # avoid no positive

    return cate_loss, reg_loss
