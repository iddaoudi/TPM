##
# =====================================================================================
#
#       Filename:  plot.py
#
#    Description:  Plot figures
#
#        Version:  1.0
#        Created:  23/03/2023
#       Revision:  none
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
    # Iterate over each group and create a separate graph for each one
    for name, group in groups:
        # Get the values for the horizontal and vertical lines
        case1 = group.loc[group['case'] == 1]
        x_val = case1['time'].iloc[0] * 1.15
        y_val = case1[['PKG1', 'PKG2', 'DRAM1', 'DRAM2']].sum(axis=1).iloc[0]
    
        # Create a scatter plot with time on the x-axis and sum of PKG1,PKG2,DRAM1,DRAM2 on the y-axis
        fig, ax = plt.subplots()
        handles = []
        for i, row in group.iterrows():
            marker_color = plt.cm.jet(colors[row['case'] - 1])
            handle = ax.scatter(row['time'], row[['PKG1', 'PKG2', 'DRAM1', 'DRAM2']].sum(), color=marker_color, marker=marker_symbols[row['case']], label='c' + str(row['case']), s=95, alpha=0.9)
            handles.append(handle)
    
        # Add the vertical and horizontal lines with legends
        ax.axvline(x_val, color='r', linestyle='--', label=f'DET + 15% ({x_val:.2f})')
        ax.axhline(y_val, color='r', linestyle='--', label=f'DEC ({y_val:.2f})')
    
        # Set the title and axis labels
        plt.title(f'{name[0]}, {name[1]}, {name[2]}')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (uJ)')
        plt.subplots_adjust(top=0.85, bottom=0.1)

        # Simplify the legend to only show the case number
        ax.legend(handles, [f'c{i}' for i in range(1, 17)], loc='center left', fancybox=True, shadow=True, bbox_to_anchor=(1, 0.5))
    
        # Save the figure and show the plot
        plt.savefig(f'function_{name[0]}_{name[1]}_{name[2]}.png', bbox_inches='tight')
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

        ax.plot(x_values, y_values, '-x', linewidth=3, color='red', label='Total energy')
        ax.axhline(y_val, color='r', linestyle='--', label='Default total energy')

        ax.legend()
        ax.grid()

        plt.title(f'{name[0]}, {name[1]}, {name[2]}')
        plt.xlabel('Cases')
        plt.ylabel('Energy consumption (uJ)')
        plt.subplots_adjust(top=0.85, bottom=0.1)

        plt.savefig(f'total_{name[0]}_{name[1]}_{name[2]}.png', bbox_inches='tight')
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

        ax.plot(x_values, y_values1, '-x', linewidth=3, color='red', label='PKG')
        ax.plot(x_values, y_values2, '-x', linewidth=3, color='blue', label='DRAM')
        ax.axhline(y_val1, color='r', linestyle='--', label='Default PKG')
        ax.axhline(y_val2, color='b', linestyle='--', label='Default DRAM')

        ax.legend()
        ax.grid()

        plt.title(f'{name[0]}, {name[1]}, {name[2]}')
        plt.xlabel('Cases')
        plt.ylabel('Energy consumption (uJ)')
        plt.subplots_adjust(top=0.85, bottom=0.1)

        plt.savefig(f'sum_{name[0]}_{name[1]}_{name[2]}.png', bbox_inches='tight')
        plt.show()

if __name__ == '__main__':
    key = int(sys.argv[1])
    df = pd.read_csv(sys.argv[2])

    groups = df.groupby(['algorithm', 'matrix_size', 'tile_size'])
    colors = np.linspace(0, 1, num=16)

    if key == 1:
        plot_function()
    elif key == 2:
        plot_total()
    elif key == 3:
        plot_sum()