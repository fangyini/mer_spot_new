import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.init as init

dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()


def jaccard_with_anchors(anchors_min, anchors_max, box_min, box_max):
    '''
    calculate tIoU for anchors and ground truth action segment
    '''
    inter_xmin = torch.max(anchors_min, box_min)
    inter_xmax = torch.min(anchors_max, box_max)
    inter_len = inter_xmax - inter_xmin

    inter_len = torch.max(inter_len, torch.tensor(0.0).type_as(dtype))
    union_len = anchors_max - anchors_min - inter_len + box_max - box_min

    jaccard = inter_len / union_len
    return jaccard


def tiou(anchors_min, anchors_max, len_anchors, box_min, box_max):
    '''
    calculate jaccatd score between a box and an anchor
    '''
    inter_xmin = np.maximum(anchors_min, box_min)
    inter_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(inter_xmax-inter_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    tiou = np.divide(inter_len, union_len)
    return tiou


def weight_init(m):
    if isinstance(m, nn.Conv1d):
        init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()


def min_max_norm(x):
    return 1.0 / (1.0 + np.exp(-1.0 * x))


def result_process_ab(video_names, video_len, start_frames, anchors_class, anchors_xmin, anchors_xmax, cfg):
    out_df = pd.DataFrame()

    xmins = np.maximum(anchors_xmin, 0)
    xmins = xmins + np.expand_dims(start_frames, axis=1)
    xmaxs = np.minimum(anchors_xmax, video_len)
    xmaxs = xmaxs + np.expand_dims(start_frames, axis=1)

    # expand video_name
    vid_name_df = list()
    num_tem_loc = anchors_xmin.shape[1]
    for i in range(len(video_names)):
        vid_names = [video_names[i]] * num_tem_loc
        vid_name_df.extend(vid_names)
    out_df['video_name'] = vid_name_df

    # reshape numpy array
    # Notice: this is not flexible enough
    num_element = anchors_xmin.shape[0] * anchors_xmin.shape[1]
    xmins_tmp = np.reshape(xmins, num_element)
    out_df['xmin'] = xmins_tmp
    xmaxs_tmp = np.reshape(xmaxs, num_element)
    out_df['xmax'] = xmaxs_tmp

    # record confidence score and category index
    scores_action = anchors_class

    if cfg.MODEL.CLS_BRANCH == False:
        max_values = np.amax(scores_action, axis=2)
        conf_tmp = np.reshape(max_values, num_element)
        out_df['conf'] = conf_tmp
        max_idxs = np.argmax(scores_action, axis=2)
    else:
        num_macro_type = int(cfg.DATASET.NUM_CLASSES / cfg.DATASET.NUM_OF_TYPE)
        num_of_type = cfg.DATASET.NUM_OF_TYPE
        macro_type = scores_action[:,:,:num_macro_type]

        max_values = np.amax(macro_type, axis=2)
        conf_tmp = np.reshape(max_values, num_element)
        out_df['conf'] = conf_tmp

        max_idxs_macro_type = np.argmax(macro_type, axis=2)
        micro_type = scores_action[:,:,num_macro_type:]

        max_idxs = np.zeros((max_idxs_macro_type.shape), dtype=int)
        for i in range(max_idxs.shape[0]):
            for j in range(max_idxs.shape[1]):
                macro_type_i = max_idxs_macro_type[i][j] # 0 macro first 4, +4 or 1 micro, last 4
                if macro_type_i == 0:
                    micro_type_i = np.argmax(micro_type[i][j][:num_of_type]) + num_of_type
                elif macro_type_i == 1:
                    micro_type_i = np.argmax(micro_type[i][j][num_of_type:])
                max_idxs[i][j] = micro_type_i
    max_idxs = max_idxs + 1
    cate_idx_tmp = np.reshape(max_idxs, num_element)
    # Notice: convert index into category type
    class_real = cfg.DATASET.CLASS_IDX
    cate_idx_tmp = np.array(class_real)[cate_idx_tmp]
    out_df['cate_idx'] = cate_idx_tmp

    out_df = out_df[cfg.TEST.OUTDF_COLUMNS_AB]

    return out_df


# Notice: maybe merge this with result_process_ab
def result_process_af(video_names, start_frames, cls_scores, anchors_xmin, anchors_xmax, cfg):
    # anchors_class,... : bs, sum_i(t_i*n_box), n_class
    # anchors_xmin, anchors_xmax: bs, sum_i(t_i*n_box)
    # video_names, start_frames: bs,
    out_df = pd.DataFrame()

    # feat_tem_width = cfg.MODEL.TEMPORAL_LENGTH[0]
    frame_window_width = cfg.DATASET.WINDOW_SIZE

    xmins = np.maximum(anchors_xmin, 0)
    xmins = xmins + np.expand_dims(start_frames, axis=1)
    # xmins = xmins / feat_tem_width * frame_window_width + np.expand_dims(start_frames, axis=1)
    xmaxs = np.minimum(anchors_xmax, frame_window_width)
    xmaxs = xmaxs + np.expand_dims(start_frames, axis=1)
    # xmaxs = xmaxs / feat_tem_width * frame_window_width + np.expand_dims(start_frames, axis=1)

    # expand video_name
    vid_name_df = list()
    num_tem_loc = anchors_xmin.shape[1]
    for i in range(len(video_names)):
        vid_names = [video_names[i]] * num_tem_loc
        vid_name_df.extend(vid_names)
    out_df['video_name'] = vid_name_df

    # reshape numpy array
    # Notice: this is not flexible enough
    num_element = anchors_xmin.shape[0] * anchors_xmin.shape[1]
    xmins_tmp = np.reshape(xmins, num_element)
    out_df['xmin'] = xmins_tmp
    xmaxs_tmp = np.reshape(xmaxs, num_element)
    out_df['xmax'] = xmaxs_tmp

    # scores_action = cls_scores[:, :, 1:]
    scores_action = cls_scores

    if cfg.MODEL.CLS_BRANCH == False:
        max_values = np.amax(scores_action, axis=2)
        conf_tmp = np.reshape(max_values, num_element)
        out_df['conf'] = conf_tmp
        max_idxs = np.argmax(scores_action, axis=2)
    else:
        num_macro_type = int(cfg.DATASET.NUM_CLASSES / cfg.DATASET.NUM_OF_TYPE)
        num_of_type = cfg.DATASET.NUM_OF_TYPE
        macro_type = scores_action[:,:,:num_macro_type]

        max_values = np.amax(macro_type, axis=2)
        conf_tmp = np.reshape(max_values, num_element)
        out_df['conf'] = conf_tmp

        max_idxs_macro_type = np.argmax(macro_type, axis=2)
        micro_type = scores_action[:,:,num_macro_type:]

        max_idxs = np.zeros((max_idxs_macro_type.shape), dtype=int)
        for i in range(max_idxs.shape[0]):
            for j in range(max_idxs.shape[1]):
                macro_type_i = max_idxs_macro_type[i][j] # 0 macro first 4, +4 or 1 micro, last 4
                if macro_type_i == 0:
                    micro_type_i = np.argmax(micro_type[i][j][:num_of_type]) + num_of_type
                elif macro_type_i == 1:
                    micro_type_i = np.argmax(micro_type[i][j][num_of_type:])
                max_idxs[i][j] = micro_type_i

    max_idxs = max_idxs + 1
    cate_idx_tmp = np.reshape(max_idxs, num_element)
    # Notice: convert index into category type
    class_real = cfg.DATASET.CLASS_IDX
    for i in range(len(cate_idx_tmp)):
        cate_idx_tmp[i] = class_real[int(cate_idx_tmp[i])]
    out_df['cate_idx'] = cate_idx_tmp

    out_df = out_df[cfg.TEST.OUTDF_COLUMNS_AB]
    return out_df
