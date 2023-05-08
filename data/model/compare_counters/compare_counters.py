import pandas as pd
import matplotlib.pyplot as plt
import sys

# default freq
df1 = pd.read_csv(sys.argv[1])
#reduced freq
df2 = pd.read_csv(sys.argv[2])

tile_sizes = df1['tile_size'].unique()
matrix_size = df1['matrix_size'].unique()
algorithm = df1['algorithm'].unique()

colors = ['red', 'blue', 'green', 'orange', 'purple']
markers_file1 = ['o', 's', 'v', '^', 'D']
markers_file2 = ['x', 'h', 'p', '>', '<']

fig, axs = plt.subplots(1, 4, figsize=(16, 6))

metrics = ['mem_boundness', 'arithm_intensity', 'ilp', 'l3_cache_ratio']

for i, metric in enumerate(metrics):
    ax = axs[i]

    for j, tile_size in enumerate(tile_sizes):
        df_tile = df1[df1['tile_size'] == tile_size]
        ax.plot(df_tile['task'], df_tile[metric], color=colors[j], marker=markers_file1[j], linestyle='-', label=f'{tile_size} - default')
        ax.scatter(df_tile['task'], df_tile[metric], color=colors[j], marker=markers_file1[j])

    for j, tile_size in enumerate(tile_sizes):
        df_tile = df2[df2['tile_size'] == tile_size]
        ax.plot(df_tile['task'], df_tile[metric], color=colors[j], marker=markers_file2[j], linestyle='-', label=f'{tile_size} - reduced')
        ax.scatter(df_tile['task'], df_tile[metric], color=colors[j], marker=markers_file2[j])

    ax.set_title(metric)
    ax.set_xlabel('Tasks')
    ax.set_ylabel(metric)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right')

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.25)

# Adjust figure margins to remove large white spaces
plt.subplots_adjust(left=0.05, right=0.95)

plt.savefig(f'compare_counters_freq_{algorithm[0]}_{matrix_size[0]}.png')
plt.show()