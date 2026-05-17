"""patient # = 3851; image#=7466"""

import csv

# read indiana_reports
read_indiana_reports = 1
print_indiana_reports = 0
if read_indiana_reports == 1:
    indiana_reports_dict = {}
    file = 'C:\multimodal_cxt\data\iu_xray\indiana_reports.csv'
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            if row[0] == 'uid':
                continue
            local_uid = row[0]
            local_dict = {'uid': row[0], 'MeSH': row[1], 'Problems': row[2], 'image': row[3], 'indication': row[4],
                          'comparison': row[5], 'findings': row[6], 'impression': row[7]}
            indiana_reports_dict[local_uid] = local_dict

    if print_indiana_reports == 1:
        for i, (k, v) in enumerate(indiana_reports_dict.items()):
            if i < 5:
                print('UID:', v['uid'])
                print('Findings:', v['findings'])
                print('Impression:', v['impression'], '\n')

# read indiana_projections
read_indiana_projections = 1
print_indiana_projections = 0
cuid = {}
if read_indiana_projections == 1:
    indiana_projections_dict = {}
    file = 'C:\multimodal_cxt\data\iu_xray\indiana_projections.csv'
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            if row[0] == 'uid':
                continue
            local_uid = row[0]  # uid,filename,projection
            local_dict = {'uid': row[0], 'filename': row[1], 'projection': row[2]}
            indiana_projections_dict[local_uid] = local_dict
            if local_uid in cuid:
                cuid[local_uid] += 1
            else:
                cuid[local_uid] = 1

    if print_indiana_projections == 1:
        for i, (k, v) in enumerate(indiana_projections_dict.items()):
            if i < 5:
                print('UID:', v['uid'])
                print('Filename:', v['filename'])
                print('Projection:', v['projection'], '\n')


# find the non-2 images patients
for uid in indiana_reports_dict:
    if uid not in indiana_projections_dict:
        print(uid)

for uid in indiana_projections_dict:
    if uid not in indiana_reports_dict:
        print(uid)

for k, v in cuid.items():
    if v != 2:
        print(k, v)