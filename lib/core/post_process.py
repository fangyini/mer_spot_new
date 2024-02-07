import pandas as pd
import os


from core.nms import temporal_nms


def get_video_fps(video_name, cfg):
    # determine FPS
    if video_name in cfg.TEST.VIDEOS_25FPS:
        fps = 25
    elif video_name in cfg.TEST.VIDEOS_24FPS:
        fps = 24
    else:
        fps = 30
    return fps


def final_result_process(out_df, epoch,subject, cfg, flag):
    '''
    flag:
    0: jointly consider out_df_ab and out_df_af
    1: only consider out_df_ab
    2: only consider out_df_af
    '''
    path_tmp = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, subject, cfg.TEST.PREDICT_TXT_FILE)
    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp)

    res_txt_file = os.path.join(path_tmp, 'test_' + str(epoch).zfill(2)+'.txt')
    res_txt_file_unnms = os.path.join(path_tmp, 'test_' + str(epoch).zfill(2) + '_unnms.txt')
    if os.path.exists(res_txt_file):
        os.remove(res_txt_file)
    
    f = open(res_txt_file, 'a')
    f_unnms = open(res_txt_file_unnms, 'a')

    if flag == 0:
        df_ab, df_af = out_df
        df_name = df_ab
    elif flag == 1:
        df_ab = out_df
        df_name = df_ab
    elif flag == 2:
        df_af = out_df
        df_name = df_af
    else:
        raise ValueError('flag should in {0, 1, 2}')

    video_name_list = list(set(df_name.video_name.values[:]))
    video_name_list.sort()

    for video_name in video_name_list:
        if flag == 0:
            df_ab, df_af = out_df
            tmpdf_ab = df_ab[df_ab.video_name == video_name]
            tmpdf_af = df_af[df_af.video_name == video_name]
            tmpdf = pd.concat([tmpdf_ab, tmpdf_af], sort=True)
        elif flag == 1:
            tmpdf = df_ab[df_ab.video_name == video_name]
        else:
            tmpdf = df_af[df_af.video_name == video_name]

        # todo: filter 842, 882, 832, 881,
        # 1421, 1463, 1421, 1464
        # 2001, 2046, 2005, 2044
        '''df1 = tmpdf[tmpdf['xmin'].between(830, 845) & tmpdf['xmax'].between(880, 883)]
        df2 = tmpdf[tmpdf['xmin'].between(1420, 1422) & tmpdf['xmax'].between(1462, 1465)]
        df3 = tmpdf[tmpdf['xmin'].between(2000, 2006) & tmpdf['xmax'].between(2043, 2047)]
        print(df1)
        print('*'*10)
        print(df2)
        print('*' * 10)
        print(df3)
        print('*' * 10)'''

        #type_set = list(set(tmpdf.cate_idx.values[:]))
        df_nms = temporal_nms(tmpdf, cfg)
        # ensure there are most 200 proposals
        df_vid = df_nms.sort_values(by='score', ascending=False)

        for i in range(len(tmpdf)):
            start_time = tmpdf.xmin.values[i]
            end_time = tmpdf.xmax.values[i]
            label = tmpdf.cate_idx.values[i]
            strout = '%s\t%.3f\t%.3f\t%d\t%.4f\n' % (
            video_name, float(start_time), float(end_time), label, tmpdf.conf.values[i])
            f_unnms.write(strout)

        for i in range(min(len(df_vid), cfg.TEST.TOP_K_RPOPOSAL)):
            start_time = df_vid.start.values[i]
            end_time = df_vid.end.values[i]
            label = df_vid.label.values[i]
            strout = '%s\t%.3f\t%.3f\t%d\t%.4f\n' % (video_name, float(start_time), float(end_time), label, df_vid.score.values[i])
            f.write(strout)
    f.close()
    f_unnms.close()
