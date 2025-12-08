import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.family'] = ['Arial', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

for atom_type in ['Li']:
    data = pd.read_pickle('./datasets/tot_data_all.pkl')
    data_atom = data[data['center_atom_type'] == atom_type].reset_index(drop=True)
    all_XAS_x = []
    all_XAS_y = []
    fig, ax = plt.subplots()
    for i in range(data_atom.shape[0]):
        atom_data = data_atom.iloc[i]
        atom_XAS = atom_data['origin_xy']
        XAS_x = [atom_XAS[i] for i in range(len(atom_XAS)) if i % 2 == 0]
        XAS_y = [atom_XAS[i] for i in range(len(atom_XAS)) if i % 2 == 1]
        # 原始数据的索引
        original_indices = np.arange(0, len(XAS_x) * 2, 2)
        # 目标索引，均匀分布在0到len(XAS_x)*2-2之间的1024个点
        target_indices = np.linspace(0, len(XAS_x) * 2 - 2, 1024)
        # 对XAS_x和XAS_y进行线性插值
        XAS_x = np.interp(target_indices, original_indices, XAS_x)
        XAS_y = np.interp(target_indices, original_indices, XAS_y)
        all_XAS_x.extend(XAS_x)
        all_XAS_y.extend(XAS_y)
    xy = np.vstack([all_XAS_x, all_XAS_y]).T
    kde = KernelDensity(bandwidth=0.1, kernel='linear', algorithm='kd_tree', leaf_size=100, rtol=1e-4, atol=1e-5).fit(xy)
    log_density = kde.score_samples(xy)
    density = np.exp(log_density)
    for cmap in ['Spectral_r', 'viridis', 'Spectral']:
        scatter = plt.scatter(all_XAS_x, all_XAS_y, c=density, cmap=cmap, zorder=100, alpha=0.2, s=1, marker='o', edgecolors='none')
        plt.colorbar(scatter, label='Density')
        ax.set_xlabel('Photon Energy(eV)', fontsize=20, fontweight='bold')
        ax.set_ylabel('Intensity(arb. units)', fontsize=20, fontweight='bold')
        ax.set_yticks([])
        ax.tick_params(axis='x', which='major', labelsize=16)
        # 在右上角添加两行文本信息，不包含文本框
        # ax.text(0.95, 0.95, f"{atom_type} K-edge\n{data_atom.shape[0]}",
        #         transform=ax.transAxes,
        #         fontsize=16,
        #         fontweight='bold',
        #         verticalalignment='top',
        #         fontfamily='Arial',
        #         horizontalalignment='right', zorder=105)
        if atom_type != 'Li':
            if atom_type == 'P':
                plt.ylim(0, 8)
            else:
                plt.ylim(0, 2.5)
        plt.savefig('kde_' + str(atom_type) + '_' + str(cmap) + '.png', dpi=1200, bbox_inches='tight', pad_inches=0.1)
        # plt.savefig('D:/caculation/xmu/XAS/feff_work/figure/spectra_feature/kde/' + 'kde_' + str(atom_type) + '_' + str(cmap) + '.pdf', dpi=1200, bbox_inches='tight', pad_inches=0.1)
        # plt.savefig('D:/caculation/xmu/XAS/feff_work/figure/spectra_feature/kde/' + 'kde_' + str(atom_type) + '_' + str(cmap) + '.svg', dpi=1200, bbox_inches='tight', pad_inches=0.1)
        # plt.savefig('D:/caculation/xmu/XAS/feff_work/figure/spectra_feature/kde/' + 'kde_' + str(atom_type) + '_' + str(cmap) + '.eps', dpi=1200, bbox_inches='tight', pad_inches=0.1)
        plt.close()
