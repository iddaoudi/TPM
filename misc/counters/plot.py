import pandas as pd
import matplotlib.pyplot as plt
import sys

df = pd.read_csv(sys.argv[1])

df_t = df.set_index('type').T
df_t.reset_index(inplace=True)

df_t.rename(columns={'index': 'metrics'}, inplace=True)

metrics1 = ['PAPI_L2_TCR', 'PAPI_L3_TCR', 'PAPI_RES_STL']
metrics2 = ['PAPI_TOT_CYC', 'PAPI_VEC_DP']

# Use a style template
plt.style.use('ggplot')

# Create a figure with larger font size
plt.rcParams.update({'font.size': 18})

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

df_t1 = df_t[df_t['metrics'].isin(metrics1)]
df_t1.plot(ax=axs[0], x='metrics', y=['compute', 'memory'], kind='bar', legend=False, color=['orange', 'skyblue'])
axs[0].set_xlabel('')
axs[0].grid(True)  # Add grid lines

df_t2 = df_t[df_t['metrics'].isin(metrics2)]
df_t2.plot(ax=axs[1], x='metrics', y=['compute', 'memory'], kind='bar', legend=False, color=['orange', 'skyblue'])
axs[1].set_xlabel('')
axs[1].grid(True)  # Add grid lines

fig.suptitle('Metrics comparison\n Data size = 40Mb')

fig.legend(['Add (compute-bound)', 'Copy (memory-bound)'], loc='upper right')

fig.subplots_adjust(top=0.1, bottom=0.05, left=0.04, right=0.05, wspace=0.05, hspace=0.05)
plt.tight_layout()

# Rotate x-tick labels
for ax in axs:
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

plt.savefig('comparison_40M.eps', format='eps')

plt.show()

