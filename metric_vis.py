import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.scale as scale
import numpy as np

fig_path = "figures/style/"
data_path = "results/formated_style.csv"
df = pd.read_csv(data_path)

datasets = ["dfs", "prim", "fractal"]

print(df.columns)

fc = df[(df["model"]=="FcVAE")&(df["is_sample"]==1)]
cn = df[(df["model"]=="ConvVAE")&(df["is_sample"]==1)]
at = df[(df["model"]=="TransformerVAE")&(df["is_sample"]==1)]
gt = df[(df["model"]=="ground_truth")&(df["is_sample"]==1)]

def plot(col, title):
    # aggr_branching_factor
    x_fc = fc[col]
    x_cn = cn[col]
    x_at = at[col]
    x_gt = gt[col]

    colors = 'Pastel1'

    X = ["Dfs", "Prim", "Fractal"]
    x_axis = np.arange(3)*2
    plt.barh(x_axis, list(x_gt), 0.4, label="Ground truth", color="#83cbeb")
    plt.barh(x_axis+.4, list(x_fc), 0.4, label="Fc VAE", color="#b4e4a2")
    plt.barh(x_axis+.8, list(x_cn), 0.4, label="Conv VAE", color="#ff7c80")
    plt.barh(x_axis+1.2, list(x_at), 0.4, label="Transformer VAE", color="#7a4983")
    plt.yticks(x_axis+0.4, X)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(fig_path+col, bbox_inches='tight')
    plt.cla()

cols = ['aggr_branching_factor','aggr_connected_components', 'aggr_count_holes_in_outer_wall',
        'aggr_has_path', 'aggr_keeps_shortest_path', 'aggr_ratio_straight_to_curl_paths']
titles = ['Branching factor','Connected Components', 'Entrances/Exits',
        'Path exists', 'Keeps Conditional', 'Straight-curl ratio']

for c, t in zip(cols, titles):
    plot(c, t)