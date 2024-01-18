import os
import numpy as np
import matplotlib.pyplot as plt
import re
import json

file = "data/data_par_t.json"

# read from json
with open(file, "r") as f:
    data = json.load(f)
    hit_rates = data["hit_rates"]
    prefetch_hit_rates = data["prefetch_hit_rates"]
    ipcs = data["ipcs"]

plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 1
plt.rcParams['grid.color'] = "#cccccc"
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.titlesize'] = 'large'
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['figure.titlesize'] = 'large'
plt.rcParams['figure.figsize'] = (9, 9)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['figure.edgecolor'] = 'black'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.edgecolor'] = 'black'
plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = 'Times New Roman'

# change the bar colors
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["#2b4251", "#5e887c", "#70a0ab", "#b6d2d9", "#fceee3"])

# bars outline
plt.rcParams['hatch.color'] = 'black'
plt.rcParams['hatch.linewidth'] = 2.0

print(hit_rates)

# {'champsim_par': {'401.bzip2-38B.champsimtrace.xz': 0.8947619435364733, '403.gcc-48B.champsimtrace.xz': 0.17985500539873514, '400.perlbench-41B.champsimtrace.xz': 0.5083242978559687, '429.mcf-22B.champsimtrace.xz': 0.37118897679100915, '444.namd-23B.champsimtrace.xz': 0.20118019888536773, '445.gobmk-30B.champsimtrace.xz': 0.4388940882857589}, 'champsim_par_base': {'403.gcc-48B.champsimtrace.xz': 0.17858798084350377, '401.bzip2-38B.champsimtrace.xz': 0.893721622780483, '445.gobmk-30B.champsimtrace.xz': 0.3856075165492206, '400.perlbench-41B.champsimtrace.xz': 0.47869697419163454, '429.mcf-22B.champsimtrace.xz': 0.29454195451022525, '444.namd-23B.champsimtrace.xz': 0.19986865148861646}}
# sort the dict by key
hit_rates["champsim_par"] = dict(sorted(hit_rates["champsim_par"].items()))
hit_rates["champsim_par_base"] = dict(sorted(hit_rates["champsim_par_base"].items()))
prefetch_hit_rates["champsim_par"] = dict(sorted(prefetch_hit_rates["champsim_par"].items()))
prefetch_hit_rates["champsim_par_base"] = dict(sorted(prefetch_hit_rates["champsim_par_base"].items()))
ipcs["champsim_par"] = dict(sorted(ipcs["champsim_par"].items()))
ipcs["champsim_par_base"] = dict(sorted(ipcs["champsim_par_base"].items()))
# three bar plots for three benchmarks
# bar plot to compare there metrics

# fig, ax = plt.subplots()
# three plots in a column
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

bar_width = 0.15
opacity = 0.8
index = np.arange(len(hit_rates["champsim_par"].keys()))

delta_hit_rates = {}
for k, v in hit_rates["champsim_par"].items():
    delta_hit_rates[k] = v - hit_rates["champsim_par_base"][k]
print(delta_hit_rates)

# ax1.bar(index, delta_hit_rates.values(), bar_width, alpha=opacity, label='LRU+AMPM+PAR', edgecolor='black')
ax1.bar(index, hit_rates["champsim_par_base"].values(), bar_width, alpha=opacity, label='LRU+AMPM', edgecolor='black')
ax1.bar(index + bar_width, hit_rates["champsim_par"].values(), bar_width, alpha=opacity, label='LRU+AMPM+PAR', edgecolor='black')

# draw the delta using o
# ax1.plot(index + bar_width * 0.5, delta_hit_rates.values(), 'o', color='black')

# ax1.set_xlabel('benchmarks')
ax1.set_ylabel('LLC hit rate')
# ax1.set_ylim(0, 1)
# ax1.set_title("LLC hit rate")
ax1.set_xticks(index + bar_width / 2)
labes = [x[:-19] for x in hit_rates["champsim_par"].keys()]
ax1.set_xticklabels(labes)

delta_prefetch_hit_rates = {}
for k, v in prefetch_hit_rates["champsim_par"].items():
    delta_prefetch_hit_rates[k] = v - prefetch_hit_rates["champsim_par_base"][k]
print(delta_prefetch_hit_rates)


# ax2.bar(index, delta_prefetch_hit_rates.values(), bar_width, alpha=opacity, label='LRU+AMPM+PAR', edgecolor='black')
ax2.bar(index, prefetch_hit_rates["champsim_par_base"].values(), bar_width, alpha=opacity, label='LRU+AMPM', edgecolor='black')
ax2.bar(index + bar_width, prefetch_hit_rates["champsim_par"].values(), bar_width, alpha=opacity, label='LRU+AMPM+PAR', edgecolor='black')

# draw the delta using o
# ax2_1.plot(index + bar_width * 0.5, delta_prefetch_hit_rates.values(), 'o', color='black')
# ax2_1.set_ylabel('delta prefetch hit rate')


# ax2.set_xlabel('benchmarks')
ax2.set_ylabel('prefetch hit rate')
# ax2.set_ylim(0, 1)
# ax2.set_title("Prefetch hit rate")

ipc_gain = {}
for k, v in ipcs["champsim_par"].items():
    ipc_gain[k] = (v - ipcs["champsim_par_base"][k]) / ipcs["champsim_par_base"][k] * 100
ax3.bar(index + bar_width * 0.5, ipc_gain.values(), bar_width, alpha=opacity, label='LRU+AMPM+PAR', edgecolor='black')

# ax3.bar(index, ipcs["champsim_par"].values(), bar_width, alpha=opacity, label='champsim_par', edgecolor='black')
# ax3.bar(index + bar_width, ipcs["champsim_par_base"].values(), bar_width, alpha=opacity, label='champsim_par_base', edgecolor='black')
# ax3.set_xlabel('benchmarks')
ax3.set_ylabel('IPC gain (%)')
# ax3.set_ylim(0, 1)
# ax3.set_title("IPC gain")


# legend
ax1.legend(loc='upper right', fontsize='medium', framealpha=0.5)
ax3.legend(loc='upper right', fontsize='medium', framealpha=0.5)

plt.tight_layout()
plt.savefig(f'plots/par_hit_rates_t.png', dpi=300)
