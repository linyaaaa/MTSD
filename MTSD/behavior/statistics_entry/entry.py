import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

parent_path = '/Volumes/Seagate Backup Plus Drive/py_vac/new_version_eprime/for_tag'

excel_fils = ['For_tag-1-1-test-mfh.xlsx', 'For_tag-1-1-test-wy.xlsx', 'For_tag-1-1-test-jt.xlsx',
       'For_tag-1-1-test-lxx.xlsx',
       'For_tag-1-1-test-ky.xlsx',  'For_tag-1-1-test-liun.xlsx', 'For_tag-1-1-test-ln.xlsx','For_tag-1-1-test-pmx.xlsx',
       'For_tag-1-1-test-llc.xlsx', 'For_tag-1-1-test-wcy.xlsx',
       'For_tag-1-1-test-lhw.xlsx', 'For_tag-1-1-test-xwy.xlsx', 'For_tag-1-1-test-zh.xlsx', 'For_tag-1-1-test-ad.xlsx', 'For_tag-1-1-test-lgs.xlsx',
        'For_tag-1-1-test-lzy.xlsx']

print('{} subjects in BEHAVIOR experiment'.format(len(excel_fils)))

keywords = ['Image', 'judge1.RESP', 'judge2.RESP', 'judge3.RESP']
record = {}

for file_index, file in enumerate(excel_fils):
    print(file)
    file_df = pd.read_excel(os.path.join(parent_path, file)).values
    name_df = list(file_df[0, :])
    key_indexes = [name_df.index(i)for i in keywords]
    target_df = file_df[1:, key_indexes]
    image_names = list(target_df[:, 0])
    data_df = np.array(target_df[:, 1:])
    for image_index, image in enumerate(image_names):
        real_image = image.split('round-')[1].split('-angel')[0]
        candidates = np.invert(np.isnan(target_df[image_index, 1:].astype(np.float)))
        # print(image_index, ', ', candidates)
        if(len(data_df[image_index,candidates]) == 0):
            pass
        else:
            label = data_df[image_index, candidates][0]
            if(len(list(record.keys())) == 0):
                record[real_image] = [(image_index, label)]
            elif(real_image in list(record.keys())):
                record[real_image].append((image_index, label))
            else:
                record[real_image] = [(image_index, label)]


mean_value = {}
for key in record.keys():
    mean_value[key] = []
    for value in record[key]:
        mean_value[key].append(value[1])

print(mean_value)
sub_value = {}
for key in mean_value.keys():
    sub_value[key] = []
    for i in range(len(excel_fils)):
        mean_ = np.mean(np.array(mean_value[key])[3*i:3*(i+1)])
        sub_value[key].append(mean_)
print(sub_value)

real_mean_value = {key: np.mean(np.array(value), axis=0) for key, value in mean_value.items()}
real_std_value = {key: np.std(np.array(value), axis=0, ddof=1) for key, value in mean_value.items()}

real_sub_mean = {key: np.mean(np.array(value), axis=0) for key, value in sub_value.items()}
real_sub_std = {key: np.std(np.array(value), axis=0) for key, value in sub_value.items()}

x_names = ['-1', '-0.9','-0.8','-0.7', '-0.6', '-0.5', '-0.4','-0.3', '-0.2','-0.1',
           '0',
           '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1',]
y_values, y_std_values = [], []
y_sub_values, y_sub_std_values = [], []
for x in x_names:
    if('-' in x):
        key_v = 'in -01'+x
        y_values.append(real_mean_value[key_v])
        y_std_values.append(real_std_value[key_v])
        y_sub_values.append(real_sub_mean[key_v])
        y_sub_std_values.append(real_sub_std[key_v])
    elif(x == '0'):
        y_values.append(0)
        y_std_values.append(0)
        y_sub_values.append(0)
        y_sub_std_values.append(0)
    else:
        key_v = 'out-01-'+x
        y_values.append(real_mean_value[key_v])
        y_std_values.append(real_std_value[key_v])
        y_sub_values.append(real_sub_mean[key_v])
        y_sub_std_values.append(real_sub_std[key_v])

print(y_values)
print(y_std_values)
print('-'*20)
print(y_sub_values)
print(y_sub_std_values)
fig = plt.figure(num=1, figsize=(15, 10))
ax = plt.gca()
ax.spines.left.set_position('zero')
ax.spines.right.set_color('none')
ax.spines.bottom.set_position('zero')
ax.spines.top.set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# line version
# plt.plot(np.arange(-1, 0, 0.1), np.array(y_values)[0:10], color='#2b8cbe', linewidth=3,
#          marker='o', markersize=10, markerfacecolor='#0868ac')
plt.plot(np.arange(-1, 0, 0.1), np.array(y_values)[0:10], color='#4d4d4d', linewidth=2.5,
         marker='o', markersize=10, markerfacecolor='#4d4d4d')
plt.plot(np.arange(0.1, 1.1, 0.1), np.array(y_values)[11:], color='#4d4d4d', linewidth=2.5,
         marker='o', markersize=10, markerfacecolor='#4d4d4d')


plt.hlines([2]*np.arange(-1, 1.1, 0.1).shape[0], xmin=-1, xmax=1, colors='#878787', linestyles='dashed')
plt.hlines([1]*np.arange(-1, 1.1, 0.1).shape[0], xmin=-1, xmax=1, colors='#878787', linestyles='dashed')
plt.fill_between(np.arange(0.55, 1.1, 0.05), y1=2.0, y2=2.5, facecolor='#fddbc7', alpha=0.5)
plt.fill_between(np.arange(-1.05, -0.9, 0.05), y1=2.0, y2=2.5, facecolor='#fddbc7', alpha=0.5)
plt.fill_between(np.arange(-0.9, 0.60, 0.05), y1=1.0, y2=2.0, facecolor='#9ecae1', alpha=0.3)
# bar version
# plt.bar(np.arange(-1, 1.1, 0.1), np.array(y_values), width=0.03, yerr=np.array(y_std_values)/2)
x_ticks_names = ['']
plt.xticks(np.arange(-1, 1.1, 0.1))
plt.ylim(0, 3)
plt.xlim((-1.1, 1.1))
degree_sign = u"\N{DEGREE SIGN}"
plt.xlabel('Disparity('+degree_sign+')', fontdict={'size':19}, labelpad=10)
plt.setp(ax.get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor', size=16)
plt.ylabel('Averaged score', fontdict={'size':19}, labelpad=400)
plt.setp(ax.get_yticklabels(), size=16)
plt.savefig('/Users/linya/Desktop/research_files/2nd/revison/'+'behavior_subs.jpg', dpi=450)
# plt.show()

# fig_2 = plt.figure(num=2, figsize=(15, 10))
# ax = plt.gca()
# ax.spines.left.set_position('zero')
# ax.spines.right.set_color('none')
# ax.spines.bottom.set_position('zero')
# ax.spines.top.set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
#
# # line version
# # plt.plot(np.arange(-1, 1.1, 0.1), y_values, color='#2b8cbe', linewidth=3,
# #          marker='o', markersize=10, markerfacecolor='#0868ac')
# # bar version
# plt.bar(np.arange(-1, 1.1, 0.1), np.array(y_sub_values), width=0.03, yerr=np.array(y_sub_std_values)/2)
# plt.xticks(np.arange(-1, 1.1, 0.1))
# plt.ylim(0, 3)
# plt.xlim((-1.1, 1.1))
# plt.show()

