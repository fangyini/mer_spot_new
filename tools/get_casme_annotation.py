import pandas as pd
import json
import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import math


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def get_casme_annotation(rawpic_path, file_path, save_file):
    rawpic_dict = {}
    max_seq_len = 0
    feat_stride=6
    num_frames=12

    subjects = [f for f in listdir(rawpic_path) if isdir(join(rawpic_path, f))]
    for s in subjects:
        videos = [f for f in listdir(join(rawpic_path, s)) if isdir(join(rawpic_path, s, f))]
        for v in videos:
            frames = [f for f in listdir(join(rawpic_path, s, v)) if isfile(join(rawpic_path, s, v, f))]
            frame_len = len(frames)
            prefix = v[:7]
            rawpic_dict[prefix] = {'name': v, 'len': frame_len}
            seq_len = (feat_stride * math.ceil((frame_len-num_frames)/feat_stride))/feat_stride
            if seq_len > max_seq_len:
                max_seq_len = seq_len
    print('max seq len=', max_seq_len)
    database = {}
    df = pd.read_excel(file_path, sheet_name=[0, 1, 2, 3, 4])
    df0 = df[0]
    df1 = df[1]
    df2 = df[2]
    df_type1 = df[3]
    df_type2 = df[4]

    # read df1 and df2 as dictionary
    df1_dict = {}
    for ind in df1.index:
        row = df1.iloc[ind]
        if row[2] not in df1_dict:
            df1_dict[row[2]] = str(row[0])
    df2_dict = {}
    for ind in df2.index:
        row = df2.iloc[ind]
        if row[1] not in df2_dict:
            df2_dict[row[1]] = '0' + str(row[0]) # change in the future
    type1_dict = {}
    for ind in df_type1.index:
        row = df_type1.iloc[ind]
        if row[0] not in type1_dict:
            type1_dict[row[0]] = row[1]
    type2_dict = {}
    for ind in df_type2.index:
        row = df_type2.iloc[ind]
        if row[0] not in type2_dict:
            type2_dict[row[0]] = row[1]

    # iterate df0
    for ind in df0.index:
        # video_name: no subset, duration(frames), annotation: [{label, segment(frames), label_id}]
        # make s15 as subset first
        row = df0.iloc[ind]
        # todo: exclude macro expression for now
        if row['Expression'] == 'macro-expression':
            continue
        sub = row['Sub']
        if sub == 1:
            subset = 'Test'
        else:
            subset = 'Train'
        name = row['Name'].split('_')[0]
        video_name = df1_dict[sub] + '_' + df2_dict[name]
        full_name = rawpic_dict[video_name]['name']
        if full_name not in database:
            database[full_name] = {} # {'subset': subset, 'duration(frames)': rawpic_dict[video_name]['len'],  'annotations': []}
        start = int(row['Onset'])
        type1 = type1_dict[row['Type1']]
        type2 = type2_dict[row['Type2']]
        database[full_name][start] = {'Type1': type1, 'Type2': type2}
        '''segment = [row['Onset'], row['Offset']]
        expression = 0 if row['Expression'] == 'macro-expression' else 1
        type1 = type1_dict[row['Type1']]
        type2 = type2_dict[row['Type2']]
        database[full_name]['annotations'].append({'segment(frames)': segment, 'expression': expression, 'type1': type1, 'type2': type2})'''
    #final_file = {"version": "casme2", "database": database}
    with open(save_file, "w") as outfile:
        json.dump(database, outfile, cls=NpEncoder)


if __name__ == '__main__':
    get_casme_annotation('/Users/adia/Documents/HKUST/micro_datasets/CAS(ME)2/rawpic/',
                         '/Users/adia/Documents/HKUST/projects/phase2/actionformer_release-main/data/casme/CAS(ME)2info.xlsx',
                         'casme2_annotation_micro.json')



