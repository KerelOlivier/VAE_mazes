import numpy as np
import matplotlib.pyplot as plt

from src.experiments.aggr_metrics import *

dfs_data = np.load('data/10k_DFS.npy', allow_pickle=True)
prim_data = np.load('data/10k_Prim.npy', allow_pickle=True)
fractal_data = np.load('data/10k_Fractal.npy', allow_pickle=True)

def plot(dfs_data, prim_data, fractal_data, metric, title, fp):
    mazes = dfs_data[:, 0]
    paths = dfs_data[:, 1]
    m_dfs = metric(mazes=mazes, paths=paths)
    mazes = prim_data[:, 0]
    paths = prim_data[:, 1]
    m_prim = metric(mazes=mazes, paths=paths)
    mazes = fractal_data[:, 0]
    paths = fractal_data[:, 1]
    m_ft = metric(mazes=mazes, paths=paths)

    x_axis = np.arange(1)*2
    plt.barh(x_axis, m_dfs, 0.2, color="#83cbeb")
    plt.barh(x_axis+.4, m_prim, 0.2, color="#ff7c80")
    plt.barh(x_axis+.8, m_ft, 0.2, color="#7a4983")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.yticks([0, 0.4, 0.8], ["DFS", "Prim", "Fractal"])
    plt.savefig(fp)



aggr_metrics = [
    ('Mean branching factor', 'figures/branch.png', aggr_branching_factor),
    ('Mean ratio straight to curled paths', 'figures/ratio_straight_curl.png', aggr_ratio_straight_to_curl_paths)
]

for metric, fp, func in aggr_metrics:
    plot(dfs_data, prim_data, fractal_data, func, metric, fp)