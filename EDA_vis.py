import numpy as np
import matplotlib.pyplot as plt

from src.experiments.aggr_metrics import *

dfs_data = np.load('data/10k_DFS.npy', allow_pickle=True)
prim_data = np.load('data/10k_Prim.npy', allow_pickle=True)
fractal_data = np.load('data/10k_Fractal.npy', allow_pickle=True)

def plot(dfs_data, prim_data, fractal_data, metric, title, fp, subset_ratio=0.01):
    dfs_data = dfs_data[:int(len(dfs_data)*subset_ratio)]
    prim_data = prim_data[:int(len(prim_data)*subset_ratio)]
    fractal_data = fractal_data[:int(len(fractal_data)*subset_ratio)]
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
    plt.yticks([0, 0.4, 0.8], ["DFS", "Prim", "Fractal"])
    plt.savefig(fp)

def count_node_degrees(maze):
    node_degrees_count = np.zeros(5)
    for i in range(1, maze.shape[0]-1):
        for j in range(1, maze.shape[1]-1):
            if maze[i, j] == 1:
                continue
            neighbors = 4 - maze[i-1, j] - maze[i+1, j] - maze[i, j-1] - maze[i, j+1]
            node_degrees_count[neighbors] += 1
    return node_degrees_count

def count_maze_degree_distribution(dataset, subset_ratio=0.01):
    dataset = dataset[:int(len(dataset)*subset_ratio)]
    mazes = dataset[:, 0]
    node_degrees = np.zeros(5)
    for i, maze in enumerate(mazes):
        node_degrees += count_node_degrees(maze)
    return node_degrees

aggr_metrics = [
    ('Mean branching factor', 'figures/branch.png', aggr_branching_factor),
    ('Mean ratio straight to curled paths', 'figures/ratio_straight_curl.png', aggr_ratio_straight_to_curl_paths)
]

for metric, fp, func in aggr_metrics:
    # continue
    plot(dfs_data, prim_data, fractal_data, func, metric, fp)

dfs_degrees = count_maze_degree_distribution(dfs_data)
prim_degrees = count_maze_degree_distribution(prim_data)
fractal_degrees = count_maze_degree_distribution(fractal_data)

normalized_dfs_degrees = dfs_degrees / np.sum(dfs_degrees)
normalized_prim_degrees = prim_degrees / np.sum(prim_degrees)
normalized_fractal_degrees = fractal_degrees / np.sum(fractal_degrees)

x_axis = np.arange(5)
plt.bar(x_axis, normalized_dfs_degrees, 0.2, color="#83cbeb")
plt.bar(x_axis+.2, normalized_prim_degrees, 0.2, color="#ff7c80")
plt.bar(x_axis+.4, normalized_fractal_degrees, 0.2, color="#7a4983")
plt.title("Node degree probability distribution")
plt.xticks(x_axis+.2, ["0", "1", "2", "3", "4"])
plt.legend(["DFS", "Prim", "Fractal"])
plt.savefig("figures/node_degree.png")
plt.show()