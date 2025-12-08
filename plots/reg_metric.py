#  encoding=utf-8
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.transforms as mtransforms

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']


def loss_function(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    metrics_dict = {
        'Epoch': [],
        'Loss': []
    }

    epoch_pattern = re.compile(r'Epoch: (\d+)')
    loss_pattern = re.compile(r'Loss: (\d+\.\d+)')

    for line in lines:
        epoch_match = epoch_pattern.search(line)
        loss_match = loss_pattern.search(line)
        if epoch_match and loss_match:
            epoch = int(epoch_match.group(1))
            loss = float(loss_match.group(1))
            metrics_dict['Epoch'].append(epoch)
            metrics_dict['Loss'].append(loss)

    return metrics_dict


OS_path = "C:/Users/zsy_s/Dropbox/xas_spectra_to_structure_project/project_simple_model/exp2/OS_dim=192_lr=0.001/result.txt"
CN_path = "C:/Users/zsy_s/Dropbox/xas_spectra_to_structure_project/project_simple_model/exp2/CN_dim=256_lr=0.001/CN_result.txt"
OCN_path = "C:/Users/zsy_s/Dropbox/xas_spectra_to_structure_project/project_simple_model/exp2/OCN_dim=256_lr=0.001/OCN_result.txt"

OS_loss = pd.DataFrame(loss_function(OS_path))
CN_loss = pd.DataFrame(loss_function(CN_path))
OCN_loss = pd.DataFrame(loss_function(OCN_path))

loss_dict = {}
loss_dict['Epoch'] = list(OS_loss['Epoch'])
loss_dict['OS'] = list(OS_loss['Loss'])
loss_dict['CN'] = list(CN_loss['Loss'])
loss_dict['OCN'] = list(OCN_loss['Loss'])

metrics_to_plot = ['OS', 'CN', 'OCN']

# 定义颜色
colors = ['tab:blue', 'tab:red', 'tab:orange']

# 创建一个新的图形
plt.figure(figsize=(10, 6))

# # 绘制每个指标的折线图
for i, metric in enumerate(metrics_to_plot):
    plt.plot(loss_dict['Epoch'], loss_dict[metric], color=colors[i], label=metric, linewidth=2)

    best_value = min(loss_dict[metric])
    best_epoch = loss_dict['Epoch'][loss_dict[metric].index(best_value)]

    offset = mtransforms.ScaledTranslation(0, 5 / 72, plt.gcf().dpi_scale_trans)
    trans = plt.gca().transData + offset
    plt.plot(best_epoch, best_value, marker='*', color='white', markersize=16, zorder=10, transform=trans)
    plt.plot(best_epoch, best_value, marker='*', color=colors[i], markersize=14, zorder=11, transform=trans)

plt.ylim(0, 1.4 * max(max(loss_dict[metric]) for metric in metrics_to_plot))
# 设置图形标题和标签
plt.title(f'Descriptor Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.legend(loc='best')

# plt.grid(True)

plt.savefig('loss.png', format='png', dpi=1600)
plt.show()


# 保存DataFrame为CSV文件
# metrics_df.to_csv('metrics.csv', index=False)