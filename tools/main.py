import os
import math
import argparse
import _init_paths
from config import cfg
from utils.utils import fix_random_seed
from config import update_config
import pprint
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from dataset.TALDataset import TALDataset
from models.fusemodel import FuseModel
from core.function import train, evaluation
from core.post_process import final_result_process
from core.utils_ab import weight_init
from utils.utils import save_model, backup_codes

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(subject, config):
    update_config(config)
    # create output directory
    if cfg.BASIC.CREATE_OUTPUT_DIR:
        out_dir = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, subject)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    # copy config file
    if cfg.BASIC.BACKUP_CODES:
        backup_dir = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, subject, 'code')
        backup_codes(cfg.BASIC.ROOT_DIR, backup_dir, cfg.BASIC.BACKUP_LISTS)
    fix_random_seed(cfg.BASIC.SEED)
    if cfg.BASIC.SHOW_CFG:
        pprint.pprint(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLE

    # check cfg values
    if cfg.DATASET.NUM_CLASSES == 2:
        assert cfg.DATASET.MICRO_SYSTEM == False
        assert cfg.MODEL.CLS_BRANCH == False
    elif cfg.DATASET.NUM_CLASSES == 8:
        assert cfg.DATASET.MICRO_SYSTEM == True

    # model
    model = FuseModel(cfg)
    model.apply(weight_init)
    model.to(DEVICE)

    if cfg.BASIC.LOAD_MODEL != 'None':
        PATH = cfg.BASIC.LOAD_MODEL + '/' + subject + '/' + 'model_' + str(cfg.BASIC.MODEL_NUM) + '.pth'
        checkpoint = torch.load(PATH, map_location=torch.device(DEVICE))
        model.load_state_dict(checkpoint['model'], strict=False)

    if cfg.BASIC.FREEZE == True:
        assert cfg.BASIC.LOAD_MODEL != 'None'
        params_to_freeze = checkpoint['model']
        for name, param in model.named_parameters():
            param.requires_grad = False if name in params_to_freeze else True

    # optimizer
    # warm_up_with_cosine_lr
    # optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=0.9, nesterov=True)
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    if cfg.TRAIN.WARM_UP_EPOCH != 0:
        warm_up_with_cosine_lr = lambda epoch: epoch / cfg.TRAIN.WARM_UP_EPOCH if epoch <= cfg.TRAIN.WARM_UP_EPOCH else 0.5 * ( math.cos((epoch - cfg.TRAIN.WARM_UP_EPOCH) /(cfg.TRAIN.END_EPOCH - cfg.TRAIN.WARM_UP_EPOCH) * math.pi) + 1)
    else:
        warm_up_with_cosine_lr = lambda \
            epoch: 0.5 * (math.cos((epoch - cfg.TRAIN.WARM_UP_EPOCH) / (cfg.TRAIN.END_EPOCH - cfg.TRAIN.WARM_UP_EPOCH) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    
    # data loader
    train_dset = TALDataset(cfg, cfg.DATASET.TRAIN_SPLIT, subject)
    train_loader = DataLoader(train_dset, batch_size=cfg.TRAIN.BATCH_SIZE,
                              shuffle=True, drop_last=False, num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.DATASET.PIN_MEMORY)
    val_dset = TALDataset(cfg, cfg.DATASET.VAL_SPLIT, subject)
    val_loader = DataLoader(val_dset, batch_size=cfg.TEST.BATCH_SIZE,
                            shuffle=False, drop_last=False, num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.DATASET.PIN_MEMORY)
    # creating log path
    log_path = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, subject, str(cfg.TRAIN.BATCH_SIZE)+'_'+cfg.TRAIN.LOG_FILE)
    if os.path.exists(log_path):
        os.remove(log_path)

    # test inference
    '''out_df_ab, out_df_af = evaluation(val_loader, model, 0, cfg)
    out_df_list = [out_df_ab, out_df_af]
    final_result_process(out_df_list, 0, subject, cfg, flag=0)
    return'''

    print('TODO: cls loss only calculate minor type clas!!')

    for epoch in range(cfg.TRAIN.END_EPOCH):
        loss_train, cls_loss_af, reg_loss_af, cls_loss_ab, reg_loss_ab, f1_ab, f1_af = train(cfg, train_loader, model, optimizer)
        print('Epoch %d: loss: %.4f AF cls loss: %.4f, reg loss: %.4f AB cls loss: %.4f, reg loss: %.4f' % (
            epoch, loss_train, cls_loss_af, reg_loss_af, cls_loss_ab, reg_loss_ab))
        print('F1 ab: ', f1_ab, ', af: ', f1_af)
        
        with open(log_path, 'a') as f:
            f.write('Epoch %d: loss: %.4f AF cls loss: %.4f, reg loss: %.4f AB cls loss: %.4f, reg loss: %.4f, f1 ab: %.4f, f1 af: %.4f \n' % (
                epoch, loss_train, cls_loss_af, reg_loss_af, cls_loss_ab, reg_loss_ab, f1_ab, f1_af))

        # decay lr
        scheduler.step()
        lr = scheduler.get_last_lr()

        if (epoch+1) % cfg.TEST.EVAL_INTERVAL == 0:
            if cfg.BASIC.SAVE_MODEL == True and epoch > 30:
                save_model(cfg, epoch=epoch, model=model, optimizer=optimizer, subject=subject)
            out_df_ab, out_df_af = evaluation(val_loader, model, epoch, cfg)
            out_df_list = [out_df_ab, out_df_af]
            final_result_process(out_df_list, epoch, subject, cfg, flag=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MER SPOT')
    parser.add_argument('--cfg', type=str, help='experiment config file', default='experiments/CAS_freeze.yaml')
    parser.add_argument('--dataset', type=str, default='cas(me)^2')
    args = parser.parse_args()

    dataset = args.dataset
    new_cfg = args.cfg

    if dataset == 'cas(me)^2':
        ca_subject = [15,16,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,35,36,37,38,40]
        ca_subject = ['s'+ str(i)  for i in ca_subject]
        for i in ca_subject:
            subject = 'subject_' + i
            print('*' * 10)
            print(subject)
            main(subject, new_cfg)
    else:
        sa_subject = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,25,26,30,32,33,34,35,36,37,99]
        sa_subject = [str(i).zfill(3)  for i in sa_subject]
        for i in sa_subject:
            subject = 'subject_' + i 
            main(subject, new_cfg)

