import os
import glob
import pandas as pd
import numpy as np
import argparse
import math
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt

def calculate_accuracy_and_f1(gt_minor_type, predicted_minor_type):
    macro_filter = np.where((gt_minor_type >= 4) & (gt_minor_type <= 6))
    if macro_filter[0].shape[0] == 0:
        f11 = 0
        cm1 = 0
    else:
        macro_gt = gt_minor_type[macro_filter]
        macro_predict = predicted_minor_type[macro_filter]
        #f11 = f1_score(macro_gt, macro_predict, average='micro')
        a = recall_score(macro_gt, macro_predict, average='macro', zero_division=0)
        b = precision_score(macro_gt, macro_predict, average='macro', zero_division=0)
        f11 = (2*a*b)/(a+b+1e-10)
        cm1 = confusion_matrix(macro_gt, macro_predict)

    micro_filter = np.where((gt_minor_type >= 1) & (gt_minor_type <= 3))
    if micro_filter[0].shape[0] == 0:
        f12 = 0
        cm2 = 0
    else:
        micro_gt = gt_minor_type[micro_filter]
        micro_predict = predicted_minor_type[micro_filter]
        #f12 = f1_score(micro_gt, micro_predict, average='micro')
        a = recall_score(micro_gt, micro_predict, average='macro', zero_division=0)
        b = precision_score(micro_gt, micro_predict, average='macro', zero_division=0)
        f12 = (2 * a * b) / (a + b + 1e-10)
        cm2 = confusion_matrix(micro_gt, micro_predict)
    return f11, f12, cm1, cm2


def max_iou(ann_csv, part_pre, TP1, TP2, write_list):
    for video_num, pre in enumerate(part_pre):
        video_name_list = list(set(ann_csv.video.values[:].tolist()))
        video_name_list.sort()

        video_name_last = part_pre[video_num][0][0]
        video_name_part = 's' + video_name_last[:2]
        video_name = os.path.join(video_name_list[0].split('/s')[0], video_name_part, video_name_last)

        video_ann_df = ann_csv[ann_csv.video == video_name]
        act_start_video = video_ann_df['startFrame'].values[:]
        # act_start_video = [sorted(i, key = lambda x:int(x)) for i in act_start_video]

        indexes = np.argsort(act_start_video)
        act_end_video = video_ann_df['endFrame'].values[:]
        act_end_video = np.array(act_end_video)[indexes]

        labels = video_ann_df['type_idx'].values[:]
        labels = np.array(labels)[indexes]

        act_start_video.sort()

        # calculate f1-score
        # number of actual frames that have been calculated so far
        pre = np.array(pre)
        pre_start = pre[:, 1].astype(float).astype(np.int64) * int(label_frequency)
        pre_end = pre[:, 2].astype(float).astype(np.int64) * int(label_frequency)

        start_tmp = list()
        end_tmp = list()
        for m in range(len(act_start_video)):
            video_label = video_name_last[:7]
            act_start = int(act_start_video[m])
            act_end = int(act_end_video[m])
            iou = (np.minimum(pre_end, act_end) - np.maximum(pre_start, act_start)) / (
                        np.maximum(pre_end, act_end) - np.minimum(pre_start, act_start))
            max_iou = np.max(iou)
            max_index = np.argmax(iou)
            if max_iou >= 0.5 and labels[m] == int(float(pre[max_index][-2])):
                tmp_write_list = [video_label, pre_start[max_index], pre_end[max_index], act_start, act_end, 'TP']
                write_list.append(tmp_write_list)
                if labels[m] == 1:
                    TP1 = TP1 + 1
                elif labels[m] == 2:
                    TP2 = TP2 + 1
                start_tmp.append(pre_start[max_index])
                end_tmp.append(pre_end[max_index])
            else:
                tmp_write_list = [video_label, '_', '_', act_start, act_end, 'FP']
                write_list.append(tmp_write_list)
        pre_start_remain = list(pre_start)
        pre_end_remain = list(pre_end)
        pre_remain_s = [i for i in pre_start_remain if i not in start_tmp]
        pre_remain_e = [i for i in pre_end_remain if i not in end_tmp]
        try:
            if len(pre_remain_s) == len(pre_remain_e):
                write_remain = [[video_label, i, pre_end[pre_start == i][0], '_', '_', 'FN'] for i in pre_remain_s]
                write_list = write_list + write_remain
            else:
                # print('lables in starts are repeat')
                write_remain = [[video_label, pre_start[pre_end == i][0], i, '_', '_', 'FN'] for i in pre_remain_e]
                write_list = write_list + write_remain
        except:
            pass
    return TP1, TP2, write_list


def all_score(TP1, TP2, N1, N2, recall1, recall2, recall_all):
    if TP1 == 0 and TP2 != 0:
        precision1 = 0
        precision2 = 1.0 * TP2 / N2
        precision_all = 1.0 * (TP1 + TP2) / (N1 + N2)
        F1_SCORE_M1 = 0
        F1_SCORE_M2 = 2 * (recall2 * precision2) / (recall2 + precision2)
        F1_SCORE = 2 * (recall_all * precision_all) / (recall_all + precision_all)
    elif TP1 != 0 and TP2 == 0:
        precision1 = 1.0 * TP1 / N1
        precision2 = 0
        precision_all = 1.0 * (TP1 + TP2) / (N1 + N2)
        F1_SCORE_M1 = 2 * (recall1 * precision1) / (recall1 + precision1)
        F1_SCORE_M2 = 0
        F1_SCORE = 2 * (recall_all * precision_all) / (recall_all + precision_all)
    elif TP1 == 0 and TP2 == 0:
        precision1 = 0
        precision2 = 0
        precision_all = 0
        F1_SCORE_M1 = 0
        F1_SCORE_M2 = 0
        F1_SCORE = 0
    else:
        precision1 = 1.0 * TP1 / N1
        precision2 = 1.0 * TP2 / N2
        precision_all = 1.0 * (TP1 + TP2) / (N1 + N2)
        F1_SCORE_M1 = 2 * (recall1 * precision1) / (recall1 + precision1)
        F1_SCORE_M2 = 2 * (recall2 * precision2) / (recall2 + precision2)
        F1_SCORE = 2 * (recall_all * precision_all) / (recall_all + precision_all)

    return F1_SCORE_M1, F1_SCORE_M2, F1_SCORE, precision_all


def main_topk(path, dataset, annotation, version):
    files_tmp = os.listdir(path)
    files = sorted(files_tmp, key=lambda x: int(x[-2:]))
    ann_csv = pd.read_csv(annotation)
    test_path_temp = [os.path.join(path, i, 'test_detection') for i in files]
    txts = glob.glob(os.path.join(test_path_temp[0], '*.txt'))

    txts = [int(i.split('_')[-1].split('.')[0]) for i in txts]
    txts.sort()
    # txt_index = txts[-1]
    best, best_m1, best_m2 = 0, 0, 0
    if dataset == 'cas(me)^2':
        out_path_tmp = os.path.join(os.path.dirname(annotation), 'top_k', 'catop_k' + '_' + str(version))
    else:
        out_path_tmp = os.path.join(os.path.dirname(annotation), 'top_k', 'satop_k' + '_' + str(version))
    if not os.path.exists(out_path_tmp):
        os.makedirs(out_path_tmp)
    best_out = os.path.join(out_path_tmp, 'best_sample.log')
    topk_out = os.path.join(out_path_tmp, 'topk.log')
    if os.path.exists(topk_out):
        os.remove(topk_out)
    for e in range(4, len(txts)):
        # for e in range(25, 26):
        txt_index = txts[e]
        test_path = [os.path.join(i, 'test_' + str(txt_index).zfill(2) + '.txt') for i in test_path_temp]
        print('number of epochs:', txt_index)
        # confirm the best top_k
        for k in range(2, 15):
            standard_out = os.path.join(out_path_tmp, 'epoch' + str(e) + '_' + str(k) + '_' + 'sample.log')
            FP, FN, TP = 0, 0, 0
            TP1, TP2 = 0, 0
            N1, N2, N_all = 0, 0, 0
            write_list = list()
            length_count = list()
            for i in test_path:
                with open(i, 'r') as f:
                    all_lines = f.readlines()
                all_lines = [h.split('\t') for h in all_lines]

                # divide predicitons of every video
                count = 1
                tmp_list = list()
                all_test = dict()
                # no duplicate label extraction
                all_video = list(set([name[0] for name in all_lines]))
                # number of GT of every video
                num_of_video = len(all_test.keys())

                for tv in all_video:
                    tmp_video = tv
                    for j in range(len(all_lines)):
                        if all_lines[j][0] == tmp_video:
                            tmp_list.append(all_lines[j])
                    all_test[count] = tmp_list
                    count = count + 1
                    tmp_list = list()
                # least len of GT
                # len_pre = [len(i) for i in all_test.values()]
                # least_len = min(len_pre)
                part_pre = [i[:k] for i in all_test.values()]

                # predictions: sorted by strat bondaries
                part_pre = [sorted(i, key=lambda x: int(float(x[1]))) for i in part_pre]

                # N1: number of precictions of macro-expressions
                # N2: number of precictions of micro-expressions
                # N_all: number of precictions
                N_all = N_all + len(part_pre) * k
                for part in part_pre:
                    N1 = N1 + len([o for o in part if o[-2] == '1'])
                    N2 = N2 + len([o for o in part if o[-2] == '2'])

                TP1, TP2, write_list = max_iou(ann_csv, part_pre, TP1, TP2, write_list)

            # calculate F1_score
            # M_all need to calculate in SAMM
            # M1： Number of macro-expressions
            # M2： Number of micro-expressions
            if dataset == 'cas(me)^2':
                M1 = 282
                M2 = 84
            else:
                M1 = 340
                M2 = 159
            recall1 = 1.0 * TP1 / M1
            recall2 = 1.0 * TP2 / M2
            recall_all = 1.0 * (TP1 + TP2) / (M1 + M2)
            F1_SCORE_M1, F1_SCORE_M2, F1_SCORE, precision_all = all_score(TP1, TP2, N1, N2, recall1, recall2,
                                                                          recall_all)
            # Sometimes, there are no predictions of micro-expressions or macro-expressions
            if F1_SCORE_M1 > best_m1:
                best_m1 = F1_SCORE_M1
                print("f1_score_macro: %05f, f1_score_micro: %05f" % (best_m1, best_m2))
            if F1_SCORE_M2 > best_m2:
                best_m2 = F1_SCORE_M2
                print("f1_score_macro: %05f, f1_score_micro: %05f" % (best_m1, best_m2))
            # record best the F1_scroe and the result of predictions
            if F1_SCORE > best:
                best = F1_SCORE
                print('number of epoch: %d, topk: %5f' % (e, k))
                print("recall: %05f, precision: %05f, f1_score: %05f" % (recall_all, precision_all, best))
                with open(best_out, 'w') as f_sout:
                    f_sout.writelines(
                        "%s, %s, %s, %s, %s, %s\n" % (wtmp[0], wtmp[1], wtmp[2], wtmp[3], wtmp[4], wtmp[5]) for wtmp in
                        write_list)
                if F1_SCORE > 0.25:
                    standard_out = os.path.join(out_path_tmp, str(e) + '_' + str(k) + '_' + 'sample.log')
                    with open(standard_out, 'w') as f_sout:
                        f_sout.writelines(
                            "%s, %s, %s, %s, %s, %s\n" % (wtmp[0], wtmp[1], wtmp[2], wtmp[3], wtmp[4], wtmp[5]) for wtmp
                            in write_list)
                    with open(topk_out, 'a') as f_threshold:
                        f_threshold.writelines("%d, %f, %d, %d, %d, %f\n" % (e, k, TP, FP, FN, F1_SCORE))
                print(TP, TP1, TP2, N1, N2)


def main_threshold(path, dataset, annotation, version, label_frequency, start_threshold, max_num_pos, k_temp):
    files_tmp = os.listdir(path)
    files = sorted(files_tmp, key=lambda x: int(x[-2:]))
    ann_csv = pd.read_csv(annotation)
    test_path_temp = [os.path.join(path, i, 'test_detection') for i in files]
    txts = glob.glob(os.path.join(test_path_temp[0], '*.txt'))

    txts = [int(i.split('_')[-1].split('.')[0]) for i in txts]
    txts.sort()

    #best, best_m1, best_m2 = 0, 0, 0
    #best_recall = 0

    final_results = {}  # epoch: subject: micro or macro: [F1 score]
    for e in range(0, 60):
        if e not in final_results.keys():
            final_results[e] = {}
        best, best_m1, best_m2 = 0, 0, 0
        best_tp1, best_tp2, best_n1, best_n2, best_f11, best_f12 = 0, 0, 0, 0, 0, 0
        cm1, cm2 = 0, 0
        best_recall = 0
        try:
            txt_index = txts[e]
        except:
            break
        # all subjects in the same epoch
        test_path = [os.path.join(i, 'test_' + str(txt_index).zfill(2) + '.txt') for i in test_path_temp]
        # confirm the best threshold

        k = 1.0 * k_temp / 1000
        FP, FN, TP = 0, 0, 0
        TP1, TP2 = 0, 0
        N1, N2, N_all = 0, 0, 0
        length_count = list()
        write_list = list()
        length_pre = list()
        predicted_minor_type = []
        gt_minor_type = []
        # every subject in one file (200x)
        for i in test_path:
            with open(i, 'r') as f:
                all_lines = f.readlines()
            all_lines = [h.split('\t') for h in all_lines]

            # divide all gts of every video
            tmp_video = all_lines[0][0]
            count = 1
            tmp_list = list()
            all_test = dict()
            all_video = list(set([name[0] for name in all_lines]))
            for tv in all_video:
                tmp_video = tv
                for j in range(len(all_lines)):
                    if all_lines[j][0] == tmp_video:
                        tmp_list.append(all_lines[j])
                all_test[count] = tmp_list
                count = count + 1
                tmp_list = list()
            # number of GT of every video
            num_of_video = len(all_test.keys())

            # least len of GT
            part_tmp = list()
            # select predictions of every video (prob > threshold)
            for i in range(num_of_video):
                tmp_one_video = list(all_test.values())[i]
                part = [o for o in tmp_one_video if float(o[-1][:-2]) > k]
                # N1: number of precictions of macro-expressions
                # N2: number of precictions of micro-expressions
                # N_all: number of precictions
                if len(part) > max_num_pos:
                    part = part[:max_num_pos]
                N_all = N_all + len(part)
                # macro label: 1, 5/6/7/8, micro: 2, 1/2/3/4
                #N1 = N1 + len([o for o in part if o[-2] == '1'])
                #N2 = N2 + len([o for o in part if o[-2] == '2'])
                N1 = N1 + len([o for o in part if o[-2] >= '4'])
                N2 = N2 + len([o for o in part if o[-2] < '4'])

                #if not part:
                #    part = [[tmp_one_video[0][0], '100000', '100000', '_', '_']]
                part_tmp.append(part)
            part_pre = part_tmp
            if len(part_tmp) == 0:
                continue

            # predictions: sorted by prob
            part_pre = [sorted(i, key=lambda x: int(float(x[1]))) for i in part_pre]

            # calculate iou between every prediction with GT
            for video_num, pre in enumerate(part_pre):
                video_name_list = list(set(ann_csv.video.values[:].tolist()))
                video_name_list.sort()
                if len(part_pre[video_num]) == 0:
                    continue

                # identify the current video
                video_name_last = part_pre[video_num][0][0]
                if dataset == 'cas(me)^2':
                    video_name_part = 's' + video_name_last[:2]
                    video_name = os.path.join(video_name_list[0].split('/s')[0], video_name_part, video_name_last)
                else:
                    video_name = os.path.join(video_name_list[0][:-4], str(video_name_last).zfill(3))

                # select startframes of current video
                video_ann_df = ann_csv[ann_csv.video == video_name]
                act_start_video = video_ann_df['startFrame'].values[:]
                # select indexes of startframes of current video
                indexes = np.argsort(act_start_video)
                # labels and endframes are sorted by indexes from actual start frames
                act_end_video = video_ann_df['endFrame'].values[:]
                act_end_video = np.array(act_end_video)[indexes]
                labels = video_ann_df['type_idx'].values[:]
                labels = np.array(labels)[indexes]
                # minor type label
                labels_minor = video_ann_df['type_idx_minor'].values[:]
                labels_minor = np.array(labels_minor)[indexes]

                # actual start frames are sorted by time series
                act_start_video.sort()

                pre = np.array(pre)
                pre_start = pre[:, 1].astype(float).astype(np.int64) * int(label_frequency)
                pre_end = pre[:, 2].astype(float).astype(np.int64) * int(label_frequency)

                #predicted_minor_type = []
                #gt_minor_type = []
                sub = int(video_name_last[:2])
                if sub not in final_results[e].keys():
                    final_results[e][sub] = {}
                    final_results[e][sub]['gt'] = []
                    final_results[e][sub]['predicted'] = []

                for m in range(len(act_start_video)):
                    act_start = int(act_start_video[m])
                    act_end = int(act_end_video[m])
                    iou = (np.minimum(pre_end, act_end) - np.maximum(pre_start, act_start)) / (
                                np.maximum(pre_end, act_end) - np.minimum(pre_start, act_start))
                    max_iou = np.max(iou)
                    max_index = np.argmax(iou)
                    if max_iou >= 0.5:
                        minor_t = int(float(pre[max_index][-2]))
                        if minor_t > 3:
                            major_t = 1
                        elif minor_t <= 3:
                            major_t = 2
                            #print(e, sub, minor_t)
                    if max_iou >= 0.5 and labels[m] == major_t:
                        # record predicted minor type
                        #predicted_minor_type.append(minor_t)
                        #gt_minor_type.append(labels_minor[m])
                        final_results[e][sub]['gt'].append(labels_minor[m])
                        final_results[e][sub]['predicted'].append(minor_t)

    plot_res = {} # s15:micro/macro:[]
    for epoch in final_results:
        for sub in final_results[epoch]:
            if sub not in plot_res:
                plot_res[sub] = np.zeros(e)
            element = final_results[epoch][sub]
            if len(element['gt']) == 0:
                continue
            gt_minor_type = np.stack(element['gt'])
            predicted_minor_type = np.stack(element['predicted'])
            macro_f1, micro_f1, cm1, cm2 = calculate_accuracy_and_f1(gt_minor_type, predicted_minor_type)
            #plot_res[sub][epoch] = micro_f1
            plot_res[sub][epoch] = macro_f1
    markers = ['o', 'v', '^', '1', 's', 'p', 'P', '*', 'H', 'x', 'X', 'D']
    keys = list(plot_res.keys())
    length = len(markers)
    x = np.arange(e)
    for i in range(length):
        sub = keys[i]
        plt.plot(x, plot_res[sub], label=sub)#, marker=markers[i])
    plt.legend()
    plt.show()
    #print()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test')

    parser.add_argument('--path', type=str, default='outputs/output_V28_downstream_freeze/cas(me)^2')
    parser.add_argument('--threshold', type=int, default=609) # macro 609, micro 341
    parser.add_argument('--ann', type=str, default='casme2_annotation_minor_type.csv')
    parser.add_argument('--dataset', type=str, default='cas(me)^2')
    parser.add_argument('--version', type=int, default=28)
    parser.add_argument('--top_k', type=bool, default=False)
    parser.add_argument('--label_frequency', type=float, default=1.0)
    parser.add_argument('--start_threshold', type=int, default=100)
    parser.add_argument('--most_pos_num', type=int, default=70)

    args = parser.parse_args()

    path = args.path
    dataset = args.dataset
    ann = args.ann
    version = args.version
    top_k = args.top_k
    label_frequency = args.label_frequency
    start_threshold = args.start_threshold
    max_num_pos = args.most_pos_num
    best_threshold = args.threshold
    if top_k:
        main_topk(path, dataset, ann, version)
    else:
        main_threshold(path, dataset, ann, version, label_frequency, start_threshold, max_num_pos, best_threshold)