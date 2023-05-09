##
# =====================================================================================
#
#       Filename:  plot.py
#
#    Description:  Plot figures
#
#        Version:  1.0
#        Created:  23/03/2023
#       Revision:  27/03/2023
#       Compiler:  python3
#
#         Author:  Idriss Daoudi <idaoudi@anl.gov>
#   Organization:  Argonne National Laboratory
#
# =====================================================================================
##

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib.gridspec as gridspec

marker_symbols = {
    1: 'o',
    2: 's',
    3: '^',
    4: 'd',
    5: '*',
    6: 'p',
    7: 'x',
    8: 'h',
    9: '+',
    10: 'v',
    11: '>',
    12: '<',
    13: 'H',
    14: 'D',
    15: 'P',
    16: 'X',
}

def plot_function():
    for name, group in groups:
        case1 = group.loc[group['case'] == 1]
        x_val = case1['time'].iloc[0] * 1.15
        y_val = case1[['PKG1', 'PKG2', 'DRAM1', 'DRAM2']].sum(axis=1).iloc[0]

        fig, ax = plt.subplots()
        handles = []
        for i, row in group.iterrows():
            marker_color = plt.cm.jet(colors[row['case'] - 1])
            handle = ax.scatter(row['time'], row[['PKG1', 'PKG2', 'DRAM1', 'DRAM2']].sum(
            ), color=marker_color, marker=marker_symbols[row['case']], label='c' + str(row['case']), s=95, alpha=0.9)
            handles.append(handle)

        ax.axvline(x_val, color='r', linestyle='--',
                   label=f'DET + 15% ({x_val:.2f})')
        ax.axhline(y_val, color='r', linestyle='--',
                   label=f'DEC ({y_val:.2f})')

        plt.title(f'{name[0]}, {name[1]}, {name[2]}')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (uJ)')
        plt.subplots_adjust(top=0.85, bottom=0.1)

        ax.legend(handles, [f'c{i}' for i in range(
            1, 17)], loc='center left', fancybox=True, shadow=True, bbox_to_anchor=(1, 0.5))

        plt.savefig(
            f'function_{name[0]}_{name[1]}_{name[2]}.png', bbox_inches='tight')
        plt.show()

def plot_total():
    for name, group in groups:
        case1 = group.loc[group['case'] == 1]
        y_val = case1[['PKG1', 'PKG2', 'DRAM1', 'DRAM2']].sum(axis=1).iloc[0]

        fig, ax = plt.subplots()

        x_values = []
        y_values = []
        for i, row in group.iterrows():
            x_values.append(row['case'])
            y_values.append(row[['PKG1', 'PKG2', 'DRAM1', 'DRAM2']].sum())

        ax.plot(x_values, y_values, '-x', linewidth=3,
                color='red', label='Total energy')
        ax.axhline(y_val, color='r', linestyle='--',
                   label='Default total energy')

        ax.legend()
        ax.grid()

        plt.title(f'{name[0]}, {name[1]}, {name[2]}')
        plt.xlabel('Cases')
        plt.ylabel('Energy consumption (uJ)')
        plt.subplots_adjust(top=0.85, bottom=0.1)

        plt.savefig(
            f'total_{name[0]}_{name[1]}_{name[2]}.png', bbox_inches='tight')
        plt.show()

def plot_sum():
    for name, group in groups:
        case1 = group.loc[group['case'] == 1]
        y_val1 = case1[['PKG1', 'PKG2']].sum(axis=1).iloc[0]
        y_val2 = case1[['DRAM1', 'DRAM2']].sum(axis=1).iloc[0]

        fig, ax = plt.subplots()

        x_values = []
        y_values1 = []
        y_values2 = []
        for i, row in group.iterrows():
            x_values.append(row['case'])
            y_values1.append(row[['PKG1', 'PKG2']].sum())
            y_values2.append(row[['DRAM1', 'DRAM2']].sum())

        ax.plot(x_values, y_values1, '-x',
                linewidth=3, color='red', label='PKG')
        ax.plot(x_values, y_values2, '-x', linewidth=3,
                color='blue', label='DRAM')
        ax.axhline(y_val1, color='r', linestyle='--', label='Default PKG')
        ax.axhline(y_val2, color='b', linestyle='--', label='Default DRAM')

        ax.legend()
        ax.grid()

        plt.title(f'{name[0]}, {name[1]}, {name[2]}')
        plt.xlabel('Cases')
        plt.ylabel('Energy consumption (uJ)')
        plt.subplots_adjust(top=0.85, bottom=0.1)

        plt.savefig(
            f'sum_{name[0]}_{name[1]}_{name[2]}.png', bbox_inches='tight')
        plt.show()

def plt_counters():
    for (algorithm, matrix_size, tile_size) in df[['algorithm', 'matrix_size', 'tile_size']].drop_duplicates().values:
        # Create a new DataFrame containing only the rows with the current matrix_size and tile_size
        df_filtered = df[(df['algorithm'] == algorithm) & (df['matrix_size'] == matrix_size) & (df['tile_size'] == tile_size)]

        tasks = df_filtered['task'].unique()
        mem_boundness = df_filtered.groupby('task')['mem_boundness'].mean()
        arithm_intensity = df_filtered.groupby('task')['arithm_intensity'].mean()
        bmr = df_filtered.groupby('task')['bmr'].mean()
        ilp = df_filtered.groupby('task')['ilp'].mean()
        l3_cache_ratio = df_filtered.groupby('task')['l3_cache_ratio'].mean()

        fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(18, 5))
        fig.suptitle(f'Matrix size = {matrix_size}, Tile size = {tile_size}')

        axs[0].bar(tasks, mem_boundness, color='r')
        axs[0].set_ylabel('Mem Boundness')

        axs[1].bar(tasks, arithm_intensity, color='g')
        axs[1].set_ylabel('Arithm Intensity')

        axs[2].bar(tasks, bmr, color='b')
        axs[2].set_ylabel('BMR')

        axs[3].bar(tasks, ilp, color='y')
        axs[3].set_ylabel('ILP')

        axs[4].bar(tasks, l3_cache_ratio, color='orange')
        axs[4].set_ylabel('L3 cache ratio')

        fig.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.savefig(f'counters_{algorithm}_{matrix_size}_{tile_size}.png')
        
def calculate_task_ratio(filename):
    df = pd.read_csv(filename)
    task_sum = df[['potrf', 'trsm', 'syrk', 'gemm']].sum(axis=1)

    ratios = df[['matrix_size', 'tile_size', 'potrf', 'trsm', 'syrk', 'gemm']].copy()
    ratios[['potrf', 'trsm', 'syrk', 'gemm']] = ratios[['potrf', 'trsm', 'syrk', 'gemm']].div(task_sum, axis=0)

    return ratios

# Using merged files
def plot_multi():
    # merged file called "final_..."
    data = pd.read_csv(sys.argv[2])
    
    matrix_sizes = data['matrix_size'].unique()
    tile_sizes = data['tile_size'].unique()
    algorithm = data['algorithm'].unique()

    activate_weights = 0
    if activate_weights == 1:
        weights = calculate_task_ratio(sys.argv[5])
    
    data['energy'] = data['PKG1'] + data['PKG2'] + data['DRAM1'] + data['DRAM2']
    data['time_energy_product'] = data['time'] * data['energy']

    for matrix_size in matrix_sizes:
        fig, axes = plt.subplots(1, len(tile_sizes), figsize=(5 * len(tile_sizes), 5), sharey=True)
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.suptitle(f'Matrix Size: {matrix_size}')

        for index, tile_size in enumerate(tile_sizes):
            if activate_weights == 1:
                # task1 = syrk = a, task2 = potrf = b, task3 = trsm = c, task4 = gemm = d
                a = weights.loc[(weights['matrix_size'] == matrix_size) & (weights['tile_size'] == tile_size), 'syrk'].values[0]
                b = weights.loc[(weights['matrix_size'] == matrix_size) & (weights['tile_size'] == tile_size), 'potrf'].values[0]
                c = weights.loc[(weights['matrix_size'] == matrix_size) & (weights['tile_size'] == tile_size), 'trsm'].values[0]
                d = weights.loc[(weights['matrix_size'] == matrix_size) & (weights['tile_size'] == tile_size), 'gemm'].values[0]
            else:
                a = b = c = d = 1
            
            data['ai'] = a*data['task1_arithm_intensity'] + b*data['task2_arithm_intensity'] + c*data['task3_arithm_intensity'] + d*data['task4_arithm_intensity']
            data['mb'] = a*data['task1_mem_boundness'] + b*data['task2_mem_boundness'] + c*data['task3_mem_boundness'] + d*data['task4_mem_boundness']
            data['ilp'] = a*data['task1_ilp'] + b*data['task2_ilp'] + c*data['task3_ilp'] + d*data['task4_ilp']
            data['cmr'] = a*data['task1_l3_cache_ratio'] + b*data['task2_l3_cache_ratio'] + c*data['task3_l3_cache_ratio'] + d*data['task4_l3_cache_ratio']
            data['sum'] = data['ai'] + data['mb'] + data['ilp'] + data['cmr']
            
            filtered_data = data[(data['matrix_size'] == matrix_size) & (data['tile_size'] == tile_size)]

            case_1_time = filtered_data.loc[filtered_data['case'] == 1, 'time'].values[0]
            case_1_energy = filtered_data.loc[filtered_data['case'] == 1, 'energy'].values[0]

            condition = (filtered_data['energy'] < case_1_energy) & (filtered_data['time'] <= case_1_time * 1.05)
            best_case = filtered_data.loc[condition].sort_values(['energy', 'time']).iloc[0]

            def color_map(row):
                if row.name == best_case.name:
                    return 'green'
                elif row['energy'] < case_1_energy and row['time'] <= case_1_time * 1.05:
                    return 'red'
                else:
                    return 'black'

            colors = filtered_data.apply(color_map, axis=1)
            metric = sys.argv[4]

            if sys.argv[3] == "product":
                ax1 = axes[index]
                ax1.scatter(filtered_data['case'], filtered_data['time_energy_product'], c=colors, marker='o')
                ax1.plot(filtered_data['case'], filtered_data['time_energy_product'], c='gray', linewidth=0.5)
                ax1.set_ylabel('Time * Energy')

                ax2 = ax1.twinx()
                ax2.scatter(filtered_data['case'], filtered_data[metric], c=colors, marker='x')
                ax2.set_ylabel(metric)
                ax2.yaxis.set_label_position('right')
                ax2.yaxis.tick_right()
                
                max_value = filtered_data[metric].max()
                min_value = filtered_data[metric].min()
                #ax2.set_ylim(min_value - 0.3, max_value + 0.3)
                
            elif sys.argv[3] == "time":
                ax1 = axes[index]
                ax1.scatter(filtered_data['case'], filtered_data['time'], c=colors, marker='o')
                ax1.plot(filtered_data['case'], filtered_data['time'], c='gray', linewidth=0.5)
                ax1.set_ylabel('Time')

                ax2 = ax1.twinx()
                ax2.scatter(filtered_data['case'], filtered_data[metric], c=colors, marker='x')
                ax2.set_ylabel(metric)
                ax2.yaxis.set_label_position('right')
                ax2.yaxis.tick_right()

            elif sys.argv[3] == "energy":
                ax1 = axes[index]
                ax1.scatter(filtered_data['case'], filtered_data['energy'], c=colors, marker='o')
                ax1.plot(filtered_data['case'], filtered_data['energy'], c='gray', linewidth=0.5)
                ax1.set_ylabel('Energy (uJ)')   

                ax2 = ax1.twinx()
                ax2.scatter(filtered_data['case'], filtered_data[metric], c=colors, marker='x')
                ax2.set_ylabel(metric)
                ax2.yaxis.set_label_position('right')
                ax2.yaxis.tick_right()

            axes[index].set_title(f'Tile Size: {tile_size}')
            axes[index].set_xlabel('Case')
            axes[index].set_xticks(filtered_data['case'])

        fig.tight_layout()
        plt.savefig(f'{sys.argv[3]}_{algorithm[0]}_{matrix_size}_{metric}.png')
        plt.show()

if __name__ == '__main__':
    key = int(sys.argv[1])
    
    if key == 1:
        df = pd.read_csv(sys.argv[2])
        groups = df.groupby(['algorithm', 'matrix_size', 'tile_size'])
        colors = np.linspace(0, 1, num=16)
        plot_function()
    elif key == 2:
        df = pd.read_csv(sys.argv[2])
        groups = df.groupby(['algorithm', 'matrix_size', 'tile_size'])
        colors = np.linspace(0, 1, num=16)
        plot_total()
    elif key == 3:
        df = pd.read_csv(sys.argv[2])
        groups = df.groupby(['algorithm', 'matrix_size', 'tile_size'])
        colors = np.linspace(0, 1, num=16)
        plot_sum()
    elif key == 4:
        df = pd.read_csv(sys.argv[2], sep=', ', engine='python')
        plt_counters()
    elif key == 5:
        plot_multi()


