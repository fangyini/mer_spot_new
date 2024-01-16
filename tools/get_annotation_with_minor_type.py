import csv
import json

annotation_file = '../casme2_annotation.csv'
new_file = '../casme2_annotation_minor_type.csv'
total_micro = 0
total_macro = 0

with open('../casme2_gt.json', 'r') as fid:
    gt_dict = json.load(fid)

test_macro = 0
test_micro = 0
for x in gt_dict:
    m1 = len(gt_dict[x]['micro']['start'])
    m2 = len(gt_dict[x]['macro']['start'])
    test_micro += m1
    test_macro += m2
print(test_macro, test_micro)

with open(new_file, 'w', newline='') as csvfile:
    #writer = csv.writer(csvfile, delimiter=' ',
    #                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer = csv.writer(csvfile)

    with open(annotation_file, "rt", encoding='ascii') as infile:
        read = csv.reader(infile)
        row_num = 0
        for row in read:
            if row[0] == 'subject':
                row.append('type_idx_minor')
            else:
                row_num += 1
                video_name = row[1].split('/')[-1]
                type = 'macro' if row[3] == '1' else 'micro' # 1 or 2
                start_frame = row[4]
                minor_type = gt_dict[video_name][type]['start'][start_frame]['Type1'] + 1 # 1, 2, 3, 4
                if minor_type == 4:
                    minor_type = -1
                else:
                    if type == 'macro':
                        minor_type += 3
                #if minor_type != -1: # and minor_type != 4:
                if type == 'macro':
                    total_macro += 1
                elif type == 'micro':
                    total_micro += 1
                row.append(minor_type)
            writer.writerow(row)

print('total macro', total_macro, ', total micro', total_micro)
print(row_num)
print('not consistent??')


