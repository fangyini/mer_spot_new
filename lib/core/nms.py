import numpy as np
import pandas as pd

from core.utils_ab import tiou

from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from itertools import takewhile

def get_items_upto_count(lst, n=1):
    data = Counter(lst).most_common()
    val = data[n-1][1] #get the value of n-1th item
    #Now collect all items whose value is greater than or equal to `val`.
    res = list(takewhile(lambda x: x[1] >= val, data))
    res = [x[0] for x in res]
    return res

def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def voting_minor_type(df, new_df, cfg, method='count'):
    # iterate every row in newdf, find the voting result with iou > 0.5:
    # by count/weighted average/weighted average after filtering low score
    # method: count, conf, countconf
    refined_label = 0
    new_df.sort_values(by='score', ascending=False, inplace=True)
    for index, row in new_df.iterrows():
        start, end = row['start'], row['end']
        duration_newdf = end-start
        original_label =int(row['label'])
        # todo: have to hardcode this
        if original_label > 3: # macro
            expression_df = df[df['cate_idx']>3]
        else:
            expression_df = df[df['cate_idx']<=3]

        start_time = np.array(expression_df.xmin.values[:])
        end_time = np.array(expression_df.xmax.values[:])
        scores = np.array(expression_df.minor_conf.values[:])
        labels = np.array(expression_df.cate_idx.values[:])

        duration = end_time - start_time

        tt1 = np.maximum(start, start_time)
        tt2 = np.minimum(end, end_time)
        intersection = tt2 - tt1
        union = (duration_newdf + duration - intersection).astype(float)
        iou = intersection / union

        inds = np.where(iou > 0.8)#[0]

        minor_types = pd.DataFrame()
        minor_types['start'] = start_time[inds]
        minor_types['end'] = end_time[inds]
        minor_types['score'] = scores[inds]
        minor_types['label'] = labels[inds]
        if method == 'count':
            final_label_list = get_items_upto_count(labels[inds].tolist())
            if len(final_label_list) == 1:
                final_label = final_label_list[0]
            else:
                if original_label in final_label_list:
                    final_label = original_label
                else:
                    final_label = final_label_list[0]
                    print('random')
        else:
            results = np.zeros(cfg.DATASET.NUM_CLASSES)
            count = np.zeros(cfg.DATASET.NUM_CLASSES)
            for _, row1 in minor_types.iterrows():
                label1 = int(row1['label'])-1
                results[label1] += row1['score']
                count[label1] += 1
            if method=='conf':
                final = results / count
            elif method=='countconf':
                final=results
            final_label=np.nanargmax(final)+1
        if final_label != original_label:
            new_df.at[index, 'label'] = final_label
            refined_label += 1
    total_per = refined_label / len(new_df)
    print('refined label percentage: ', total_per)
    return new_df


def temporal_nms(df, cfg):
    '''
    temporal nms
    I should understand this process
    '''

    type_set = list(set(df.cate_idx.values[:]))
    if cfg.DATASET.NUM_CLASSES > 2:
        type_set_new = [[], []]
        for i in type_set:
            if i <= cfg.DATASET.NUM_OF_TYPE:
                type_set_new[0].append(i)
            else:
                type_set_new[1].append(i)
        type_set = type_set_new
    # type_set.sort()

    # returned values
    rstart = list()
    rend = list()
    rscore = list()
    rlabel = list()

    # attention: for THUMOS, a sliding window may contain actions from different class
    for t in type_set:
        if cfg.DATASET.NUM_CLASSES <= 2:
            #label = t
            tmp_df = df[df.cate_idx == t]
        else:
            if len(t) == 0:
                continue
            out_df = pd.DataFrame()
            for i in t:
                tmp_df = df[df.cate_idx == i]
                out_df = pd.concat([out_df, tmp_df])
            tmp_df = out_df

        tmp_df = tmp_df.sort_values(by='xmax', ascending=False)
        tmp_df = tmp_df.sort_values(by='xmin', ascending=False)
        label = np.array(tmp_df.cate_idx.values[:])

        start_time = np.array(tmp_df.xmin.values[:])
        end_time = np.array(tmp_df.xmax.values[:])
        scores = np.array(tmp_df.conf.values[:])

        duration = end_time - start_time
        order = scores.argsort()[::-1]

        keep = list()
        while (order.size > 0) and (len(keep) < cfg.TEST.TOP_K_RPOPOSAL):
            i = order[0]
            keep.append(i)
            tt1 = np.maximum(start_time[i], start_time[order[1:]])
            tt2 = np.minimum(end_time[i], end_time[order[1:]])
            intersection = tt2 - tt1
            union = (duration[i] + duration[order[1:]] - intersection).astype(float)
            iou = intersection / union

            inds = np.where(iou <= cfg.TEST.NMS_TH)[0]
            order = order[inds + 1]

        # record the result
        for idx in keep:
            #if cfg.DATASET.NUM_CLASSES > 2:
            rlabel.append(label[idx])
            #else:
            #    rlabel.append(label)
            rstart.append(float(start_time[idx]))
            rend.append(float(end_time[idx]))
            rscore.append(scores[idx])

    new_df = pd.DataFrame()
    new_df['start'] = rstart
    new_df['end'] = rend
    new_df['score'] = rscore
    new_df['label'] = rlabel

    #if cfg.MODEL.CLS_BRANCH == True:
    #    new_df = voting_minor_type(df, new_df, cfg)
    return new_df


def soft_nms(df, idx_name, cfg):
    df = df.sort_values(by='score', ascending=False)
    save_file = '/data/home/v-yale/ActionLocalization/output/df_sort.csv'
    df.to_csv(save_file, index=False)

    tstart = list(df.start.values[:])
    tend = list(df.end.values[:])
    tscore = list(df.score.values[:])
    tcls_type = list(df.cls_type.values[:])
    rstart = list()
    rend = list()
    rscore = list()
    rlabel = list()

    while len(tscore) > 0 and len(rscore) <= cfg.TEST.TOP_K_RPOPOSAL:
        max_idx = np.argmax(tscore)
        tmp_width = tend[max_idx] - tstart[max_idx]
        iou = tiou(tstart[max_idx], tend[max_idx], tmp_width, np.array(tstart), np.array(tend))
        iou_exp = np.exp(-np.square(iou) / cfg.TEST.SOFT_NMS_ALPHA)
        for idx in range(len(tscore)):
            if idx != max_idx:
                tmp_iou = iou[idx]
                threshold = cfg.TEST.SOFT_NMS_LOW_TH + (cfg.TEST.SOFT_NMS_HIGH_TH - cfg.TEST.SOFT_NMS_LOW_TH) * tmp_width
                if tmp_iou > threshold:
                    tscore[idx] = tscore[idx] * iou_exp[idx]
        rstart.append(tstart[max_idx])
        rend.append(tend[max_idx])
        rscore.append(tscore[max_idx])
        # video class label
        cls_type = tcls_type[max_idx]
        label = idx_name[cls_type]
        rlabel.append(label)

        tstart.pop(max_idx)
        tend.pop(max_idx)
        tscore.pop(max_idx)
        tcls_type.pop(max_idx)

    new_df = pd.DataFrame()
    new_df['start'] = rstart
    new_df['end'] = rend
    new_df['score'] = rscore
    new_df['label'] = rlabel
    return new_df

