import torch
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_f1_score, multiclass_precision, multiclass_recall
from torchmetrics.classification import Recall, Precision

dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
precision_metric = Precision(task="multiclass", average='macro', num_classes=3).to(DEVICE)
recall_metric = Recall(task="multiclass", average='macro', num_classes=3).to(DEVICE)

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


class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=20, eps=1e-6, class_weight=None):
        super(Focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.eps = eps
        self.class_weight = torch.tensor(class_weight).type_as(dtype) if class_weight is not None else None

    def forward(self, x, y): #af_cls: cls_af + cls_af_type
        t = one_hot_embedding(y, 1 + self.num_classes)
        t = t[:, 1:]
        #t = one_hot_embedding(y, self.num_classes)

        p = x.sigmoid()
        pt = p * t + (1 - p) * (1 - t)  # pt = p if t > 0 else 1-p
        pt = pt.clamp(min=self.eps)  # avoid log(0)
        w = self.alpha * t + (1 - self.alpha) * (1 - t)  # w = alpha if t > 0 else 1-alpha
        loss = -(w * (1 - pt).pow(self.gamma) * torch.log(pt))
        loss = loss.sum(dim=0)
        if self.class_weight is not None:
            loss = self.class_weight * loss
        return loss.sum()


def loss_function_ab(anchors_x, anchors_w, anchors_rx_ls, anchors_rw_ls, anchors_class,
                     match_x, match_w, match_scores, match_labels, cfg, weight, check):
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

    if cfg.MODEL.CLS_BRANCH == False:
        anchors_class = anchors_class.view(-1, cfg.DATASET.NUM_CLASSES)[keep]
    else:
        anchors_class = anchors_class.view(-1, cfg.DATASET.NUM_CLASSES+int(cfg.DATASET.NUM_CLASSES/cfg.DATASET.NUM_OF_TYPE))[keep]

    match_labels = match_labels.view(-1)[keep]
    check = check.view(-1)[keep]

    if cfg.MODEL.CLS_BRANCH == False:
        cate_loss_f = Focal_loss(num_classes=cfg.DATASET.NUM_CLASSES, class_weight=weight)
        cls_loss = cate_loss_f(anchors_class, match_labels)
        f1 = 0
    else:
        major_type = int(cfg.DATASET.NUM_CLASSES/cfg.DATASET.NUM_OF_TYPE)
        minor_type = cfg.DATASET.NUM_OF_TYPE
        # cate_label to only 1 and 2
        ind_micro = torch.where((match_labels<=minor_type) & (match_labels>0))
        ind_macro = torch.where(match_labels>minor_type)
        major_cate_label = match_labels.clone()
        major_cate_label[ind_micro] = 2
        major_cate_label[ind_macro] = 1
        cate_loss_f = Focal_loss(num_classes=major_type, class_weight=weight)
        cls_loss_exp = cate_loss_f(anchors_class[:, :major_type], major_cate_label)

        # only use positive sample,
        pred_cls_minor = anchors_class[:, major_type:]

        # print check
        '''checking_index = torch.where(check > 0)
        pred_checking = pred_cls_minor[checking_index][:, 3:]
        pred_checking = torch.argmax(pred_checking, dim=1)  # should be all 0
        if pred_checking.size()[0] != 0:
            acc = len(torch.where(pred_checking == 0)[0]) / pred_checking.size()[0]
            print('AB pred acc:', acc)'''

        pos_sample = torch.where(match_labels>0)
        if pos_sample[0].size()[0] == 0:
            cls_loss_type = torch.tensor(0).to(DEVICE)
            f1=torch.nan
        else:
            cate_label_minor = match_labels[pos_sample]
            pred_cls_minor = pred_cls_minor[pos_sample]

            me_labels_ind = torch.where(cate_label_minor <= minor_type)
            me_label_gt = cate_label_minor[me_labels_ind] - 1  # me: 1,2,3 -> 0,1,2
            me_label_pred = pred_cls_minor[me_labels_ind][:, :minor_type]
            if me_labels_ind[0].size()[0] == 0:
                cls_loss_me = torch.tensor(0).to(DEVICE)
            else:
                me_weight = torch.tensor([27, 73, 35]).to(DEVICE)
                me_weight = me_weight.sum() / me_weight
                ce_loss_me = nn.CrossEntropyLoss(weight=me_weight)
                cls_loss_me = ce_loss_me(me_label_pred, me_label_gt)

            mae_labels_ind = torch.where(cate_label_minor > minor_type)
            mae_label_gt = cate_label_minor[mae_labels_ind] - 1 - minor_type  # mae: 4,5,6->0,1,2
            mae_label_pred = pred_cls_minor[mae_labels_ind][:, minor_type:]
            if mae_labels_ind[0].size()[0] == 0:
                cls_loss_mae = torch.tensor(0).to(DEVICE)
            else:
                mae_weight = torch.tensor([304, 288, 53]).to(DEVICE)
                mae_weight = mae_weight.sum() / mae_weight
                ce_loss_mae = nn.CrossEntropyLoss(weight=mae_weight)
                cls_loss_mae = ce_loss_mae(mae_label_pred, mae_label_gt)

            cls_loss_type = cls_loss_me + cls_loss_mae
            with torch.no_grad():  # change back to 1 to 6
                final_pred_me = torch.argmax(me_label_pred, dim=1)
                final_pred_mae = torch.argmax(mae_label_pred, dim=1) + minor_type
                final_pred = torch.cat((final_pred_me, final_pred_mae))
                final_label = torch.cat((me_label_gt, mae_label_gt + minor_type))
                f1 = multiclass_f1_score(final_pred, final_label, average='micro')
                f1 = f1.cpu().numpy()
        #cls_loss = cls_loss_exp + cls_loss_type * 10
        cls_loss = cls_loss_type
    '''if torch.sum(pmask) > 0:
        cls_loss = cls_loss / torch.sum(pmask)  # avoid no positive
    else:
        cls_loss = torch.tensor(0.).type_as(cls_loss)'''

    # localization loss
    if torch.sum(pmask) > 0:
        keep = pmask
        target_rx = target_rx.view(-1)[keep]
        target_rw = target_rw.view(-1)[keep]
        anchors_rx_ls = anchors_rx_ls.view(-1)[keep]
        anchors_rw_ls = anchors_rw_ls.view(-1)[keep]

        loc_loss = abs_smooth(target_rx - anchors_rx_ls) + abs_smooth(target_rw - anchors_rw_ls)
    else:
        loc_loss = torch.tensor(0.).to(DEVICE) #type_as(cls_loss)
    # print('loss:', cls_loss.item(), loc_loss.item(), overlap_loss.item())

    return cls_loss, loc_loss, f1


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


def loss_function_af(cate_label, preds_cls, target_loc, pred_loc, cfg, weight, checking_label):
    '''
    preds_cls: bs, t1+t2+..., n_class
    pred_regs_batch: bs, t1+t2+..., 2
    '''
    batch_size = preds_cls.size(0)
    cate_label_view = cate_label.view(-1)
    cate_label_view = cate_label_view.type_as(dtypel)
    checking_view = checking_label.view(-1)
    checking_view = checking_view.type_as(dtypel)
    if cfg.MODEL.CLS_BRANCH == False:
        preds_cls_view = preds_cls.view(-1, cfg.DATASET.NUM_CLASSES)
    else:
        preds_cls_view = preds_cls.view(-1, cfg.DATASET.NUM_CLASSES+int(cfg.DATASET.NUM_CLASSES/cfg.DATASET.NUM_OF_TYPE))
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
    if cfg.MODEL.CLS_BRANCH == False:
        cate_loss_f = Focal_loss(num_classes=cfg.DATASET.NUM_CLASSES, class_weight=weight)
        cls_loss = cate_loss_f(preds_cls_view, cate_label_view)
        f1 = 0
    else:
        major_type = int(cfg.DATASET.NUM_CLASSES/cfg.DATASET.NUM_OF_TYPE)
        minor_type = cfg.DATASET.NUM_OF_TYPE
        # cate_label to only 1 and 2
        ind_micro = torch.where((cate_label_view<=minor_type) & (cate_label_view>0))
        ind_macro = torch.where(cate_label_view>minor_type)
        major_cate_label = cate_label_view.clone()
        major_cate_label[ind_micro] = 2
        major_cate_label[ind_macro] = 1
        cate_loss_f = Focal_loss(num_classes=major_type, class_weight=weight)
        cls_loss_exp = cate_loss_f(preds_cls_view[:, :major_type], major_cate_label)

        # only use positive sample
        pred_cls_minor = preds_cls_view[:, major_type:]

        # checking
        '''checking_index = torch.where(checking_view>0)
        #gt_checking = cate_label_view[checking_index]
        #assert gt_checking.any() == 4
        pred_checking = pred_cls_minor[checking_index][:, 3:]
        pred_checking = torch.argmax(pred_checking, dim=1) # should be all 0
        if pred_checking.size()[0] != 0:
            acc = len(torch.where(pred_checking == 0)[0]) / pred_checking.size()[0]
            print('AF pred acc:', acc)'''

        pos_sample = torch.where(cate_label_view>0)
        if pos_sample[0].size()[0] == 0:
            cls_loss_type = torch.tensor(0).to(DEVICE)
            f1 = torch.nan
        else:
            cate_label_minor = cate_label_view[pos_sample]
            pred_cls_minor = pred_cls_minor[pos_sample]

            me_labels_ind = torch.where(cate_label_minor<=minor_type)
            me_label_gt = cate_label_minor[me_labels_ind] - 1  # me: 1,2,3 -> 0,1,2
            me_label_pred = pred_cls_minor[me_labels_ind][:, :minor_type]
            if me_labels_ind[0].size()[0] == 0:
                cls_loss_me = torch.tensor(0).to(DEVICE)
            else:
                me_weight = torch.tensor([27, 73, 35]).to(DEVICE)
                me_weight = me_weight.sum() / me_weight
                ce_loss_me = nn.CrossEntropyLoss(weight=me_weight)
                cls_loss_me = ce_loss_me(me_label_pred, me_label_gt)

            mae_labels_ind = torch.where(cate_label_minor>minor_type)
            mae_label_gt = cate_label_minor[mae_labels_ind] - 1 - minor_type  # mae: 4,5,6->0,1,2
            mae_label_pred = pred_cls_minor[mae_labels_ind][:, minor_type:]
            if mae_labels_ind[0].size()[0] == 0:
                cls_loss_mae = torch.tensor(0).to(DEVICE)
            else:
                mae_weight = torch.tensor([304, 288, 53]).to(DEVICE)
                mae_weight = mae_weight.sum() / mae_weight
                ce_loss_mae = nn.CrossEntropyLoss(weight=mae_weight)
                cls_loss_mae = ce_loss_mae(mae_label_pred, mae_label_gt)
            cls_loss_type = cls_loss_me + cls_loss_mae
            with torch.no_grad(): # change back to 1 to 6
                final_pred_mae = torch.argmax(mae_label_pred, dim=1) + minor_type
                final_pred_me = torch.argmax(me_label_pred, dim=1)
                final_pred = torch.cat((final_pred_me, final_pred_mae))
                final_label = torch.cat((me_label_gt, mae_label_gt + minor_type))
                f1 = multiclass_f1_score(final_pred, final_label, average='micro')
                f1 = f1.cpu().numpy()
            '''final_pred = []
            final_label = []
            for i in range(pred_cls_minor.size()[0]):
                l = cate_label_minor[i]
                if l <= minor_type: # micro, last 4
                    final_pred.append(pred_cls_minor[i][minor_type:])
                else: # macro, first 4
                    final_pred.append(pred_cls_minor[i][:minor_type])
                    l -= minor_type
                final_label.append(l)
            final_pred = torch.stack(final_pred) # 0, 1, 2
            final_label = torch.stack(final_label) - 1 # 1, 2, 3 - 1 = 0, 1, 2
            ce_loss = nn.CrossEntropyLoss()
            cls_loss_type = ce_loss(final_pred, final_label)
            with torch.no_grad():
                final_pred = torch.argmax(final_pred, dim=1)
                precision = precision_metric(final_pred, final_label)
                recall = recall_metric(final_pred, final_label)

                f1 = 2*precision*recall / (precision+recall)
                f1 = f1.cpu().numpy()'''
                #print('f1 score: ', f1)
            #cate_loss_f_type = Focal_loss(num_classes=minor_type, class_weight=weight)
            #cls_loss_type = cate_loss_f_type(final_pred, final_label)
        #cls_loss = cls_loss_exp + cls_loss_type * 10  # todo: check range
        cate_loss = cls_loss_type

    '''if torch.sum(pmask) > 0:
        cate_loss = cls_loss / torch.sum(pmask)  # avoid no positive
    else:
        cate_loss = torch.tensor(0.).to(DEVICE)''' #.type_as(cls_loss)

    #cate_loss = cls_loss / (torch.sum(pmask) + batch_size)  # avoid no positive
    return cate_loss, reg_loss, f1
