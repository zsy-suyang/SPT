#  encoding=utf-8
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']

task = "CN"
path = "C:/Users/zsy_s/Dropbox/xas_spectra_to_structure_project/project_simple_model/exp2/CN_dim=256_lr=0.001/CN_result.txt"
with open(path, 'r') as file:
    lines = file.readlines()

# 初始化一个字典来存储指标
metrics_dict = {
    'Epoch': [],
    'Loss': [],
    'Pre': [],
    'Rec': [],
    'F1': [],
    'ACC': [],
    'AUC': []
}

# 定义一个正则表达式模式来匹配指标
epoch_pattern = re.compile(r'Epoch: (\d+)')
loss_pattern = re.compile(r'Loss: (\d+\.\d+)')
precision_pattern = re.compile(r'Precison: (\d+\.\d+)')
recall_pattern = re.compile(r'Rec: (\d+\.\d+)')
f1_pattern = re.compile(r'F1: (\d+\.\d+)')
acc_pattern = re.compile(r'acc: (\d+\.\d+)')
auc_pattern = re.compile(r'auc: (\d+\.\d+)')

# 遍历每一行，提取指标并存储到字典中
for line in lines:
    epoch_match = epoch_pattern.search(line)
    loss_match = loss_pattern.search(line)
    precision_match = precision_pattern.search(line)
    recall_match = recall_pattern.search(line)
    f1_match = f1_pattern.search(line)
    acc_match = acc_pattern.search(line)
    auc_match = auc_pattern.search(line)

    if epoch_match and loss_match and precision_match and recall_match and f1_match and acc_match and auc_match:
        epoch = int(epoch_match.group(1))
        loss = float(loss_match.group(1))
        precision = float(precision_match.group(1))
        recall = float(recall_match.group(1))
        f1 = float(f1_match.group(1))
        acc = float(acc_match.group(1))
        auc = float(auc_match.group(1))

        metrics_dict['Epoch'].append(epoch)
        metrics_dict['Loss'].append(loss)
        metrics_dict['Pre'].append(precision)
        metrics_dict['Rec'].append(recall)
        metrics_dict['F1'].append(f1)
        metrics_dict['ACC'].append(acc)
        metrics_dict['AUC'].append(auc)


metrics_df = pd.DataFrame(metrics_dict)

metrics_to_plot = ['Pre', 'Rec', 'F1', 'ACC', 'AUC']

# 定义颜色
colors = ['tab:purple', 'tab:green', 'tab:blue', 'tab:red', 'tab:orange']

# 创建一个新的图形
plt.figure(figsize=(10, 6))

# 绘制每个指标的折线图
for i, metric in enumerate(metrics_to_plot):
    plt.plot(metrics_df['Epoch'], metrics_df[metric], color=colors[i], label=metric, linewidth=1.5)

    best_value = metrics_df[metric].max()
    best_epoch = metrics_df.loc[metrics_df[metric].idxmax()]['Epoch']

    plt.plot(best_epoch, best_value, marker='*', color=colors[i], markersize=16)

# 设置图形标题和标签
plt.title(f'Descriptor {task} Metrics', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.legend(loc='best')

# # 显示网格线
# plt.grid(True)

plt.savefig(f'{task}_metrics.png', format='png', dpi=1600)


# 显示图形
plt.show()


# 保存DataFrame为CSV文件
# metrics_df.to_csv('metrics.csv', index=False)