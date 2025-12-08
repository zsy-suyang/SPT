import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
import matplotlib.ticker as mtick
import matplotlib.font_manager as fm

font_path = '/internfs/zsy/proj/AMFormer/plots/Arial.ttf'

try:
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    custom_font_name = prop.get_name()
    plt.rcParams['font.family'] = custom_font_name
    print(f"成功加载字体文件: {font_path}")
    print(f"全局字体已设置为: {custom_font_name}")
    
except Exception as e:
    print(f"加载字体失败，将使用默认字体。错误信息: {e}")
    plt.rcParams['font.family'] = 'sans-serif'

tasks = ['OS', 'OCN', 'CN', 'CN2', 'PSGO', 'NNRS', 'MOOD'][::-1]
expert_scores = np.array([0.9679, 0.9820, 0.9814, 0.8110, 0.5051, 0.7519, 0.6700][::-1])
joint_scores = np.array([0.9582, 0.9779, 0.9756, 0.8054, 0.5129, 0.7336, 0.6801][::-1])
diffs = joint_scores - expert_scores
diffs_pct = diffs * 100 
n_reg = 2

plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.0

c_expert = "#336d9c"  
c_joint = "#7b1719"   
c_gray_line = '#b3b3b3'

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5.5), 
                               gridspec_kw={'width_ratios': [1.3, 1], 'wspace': 0.1})

y_pos = np.arange(len(tasks))

for y, e, j in zip(y_pos, expert_scores, joint_scores):
    ax1.plot([e, j], [y, y], color=c_gray_line, lw=1.5, zorder=1, alpha=0.6)

ax1.scatter(expert_scores, y_pos, color=c_expert, label='Expert SPT', s=100, zorder=3)
ax1.scatter(joint_scores, y_pos, color=c_joint, label='Multi-Target Joint SPT', s=100, zorder=3)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(tasks, fontweight='bold', fontsize=11)
ax1.set_xlabel('F1-Score or R2', fontweight='bold', fontsize=12)
ax1.set_xlim(0.45, 1.05) 
ax1.grid(axis='x', linestyle='--', alpha=0.4)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.tick_params(axis='y', length=0) 

legend_handles = [
    mlines.Line2D([], [], color=c_expert, marker='o', linestyle='None', markersize=9, label='Expert SPT'),
    mlines.Line2D([], [], color=c_joint, marker='o', linestyle='None', markersize=9, label='Multi-Target Joint SPT')
]
ax1.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(-0.15, 1.12), 
           ncol=2, frameon=False, fontsize=11)

colors_bar = [c_joint if d > 0 else c_expert for d in diffs_pct]
bars = ax2.barh(y_pos, diffs_pct, height=0.6, color=colors_bar, alpha=0.85)
ax2.axvline(0, color='black', linewidth=0.8, linestyle='-')

for bar, d_pct in zip(bars, diffs_pct):
    width = bar.get_width()
    label_x = width + (0.15 if d_pct > 0 else -0.15) 
    ha_align = 'left' if d_pct > 0 else 'right'
    label_text = f"{d_pct:+.2f}%"
    ax2.text(label_x, bar.get_y() + bar.get_height()/2, label_text, 
             va='center', ha=ha_align, fontsize=10, fontweight='medium', 
             color='#333333')

ax2.set_xlabel('Performance Gap', fontweight='bold', fontsize=12)
ax2.set_yticks([]) 
max_limit = max(abs(diffs_pct)) * 1.6 
ax2.set_xlim(-max_limit, max_limit)

ax2.xaxis.set_major_formatter(mtick.PercentFormatter())
ax2.grid(axis='x', linestyle='--', alpha=0.4)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_position(('data', -0.6))

sep_y = n_reg - 0.5 
ax1.axhline(sep_y, color='gray', linestyle=':', linewidth=1, alpha=0.5, xmin=-0.1, clip_on=False)
ax2.axhline(sep_y, color='gray', linestyle=':', linewidth=1, alpha=0.5, clip_on=False)

ax1.text(0.46, 0.5, 'Regression', fontsize=10, color='gray', style='italic', va='center')
ax1.text(0.46, 3.5, 'Classification', fontsize=10, color='gray', style='italic', va='center')

fig.text(0.9, 0.92, 'Dataset: non-oxygen', 
         ha='right', va='bottom', 
         fontsize=12, fontweight='bold', color='#333333')

plt.tight_layout()
plt.subplots_adjust(top=0.88)

save_path = 'comparison_chart_arial.svg'
plt.savefig(save_path, format='svg', dpi=600, bbox_inches='tight')
print(f"图表已保存至: {save_path}")
plt.show()