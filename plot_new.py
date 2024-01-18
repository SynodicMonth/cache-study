import os
import numpy as np
import matplotlib.pyplot as plt
import re
import json

file = "data/445.gobmk-30B.champsimtrace.xz.json"

# read from json
with open(file, "r") as f:
    data = json.load(f)
    hit_rates = data["hit_rates"]
    prefetch_hit_rates = data["prefetch_hit_rates"]

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
plt.rcParams['figure.figsize'] = (9, 6)
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

lcc_hit_rates_formatted = {"no": {}, "nextline": {}, "ipstride": {}, "sppdev": {}, "vaampm": {}}
prefetch_hit_rates_formatted = {"no": {}, "nextline": {}, "ipstride": {}, "sppdev": {}, "vaampm": {}}

for key in hit_rates.keys():
    if "no" in key:
        lcc_hit_rates_formatted["no"][key.split("_")[1]] = hit_rates[key]
        prefetch_hit_rates_formatted["no"][key.split("_")[1]] = prefetch_hit_rates[key]
    elif "nextline" in key:
        lcc_hit_rates_formatted["nextline"][key.split("_")[1]] = hit_rates[key]
        prefetch_hit_rates_formatted["nextline"][key.split("_")[1]] = prefetch_hit_rates[key]
    elif "ipstride" in key:
        lcc_hit_rates_formatted["ipstride"][key.split("_")[1]] = hit_rates[key]
        prefetch_hit_rates_formatted["ipstride"][key.split("_")[1]] = prefetch_hit_rates[key]
    elif "sppdev" in key:
        lcc_hit_rates_formatted["sppdev"][key.split("_")[1]] = hit_rates[key]
        prefetch_hit_rates_formatted["sppdev"][key.split("_")[1]] = prefetch_hit_rates[key]
    elif "vaampm" in key:
        lcc_hit_rates_formatted["vaampm"][key.split("_")[1]] = hit_rates[key]
        prefetch_hit_rates_formatted["vaampm"][key.split("_")[1]] = prefetch_hit_rates[key]

# sort the replacements by name
lcc_hit_rates_formatted["no"] = dict(sorted(lcc_hit_rates_formatted["no"].items()))
lcc_hit_rates_formatted["nextline"] = dict(sorted(lcc_hit_rates_formatted["nextline"].items()))
lcc_hit_rates_formatted["ipstride"] = dict(sorted(lcc_hit_rates_formatted["ipstride"].items()))
lcc_hit_rates_formatted["sppdev"] = dict(sorted(lcc_hit_rates_formatted["sppdev"].items()))
lcc_hit_rates_formatted["vaampm"] = dict(sorted(lcc_hit_rates_formatted["vaampm"].items()))

prefetch_hit_rates_formatted["no"] = dict(sorted(prefetch_hit_rates_formatted["no"].items()))
prefetch_hit_rates_formatted["nextline"] = dict(sorted(prefetch_hit_rates_formatted["nextline"].items()))
prefetch_hit_rates_formatted["ipstride"] = dict(sorted(prefetch_hit_rates_formatted["ipstride"].items()))
prefetch_hit_rates_formatted["sppdev"] = dict(sorted(prefetch_hit_rates_formatted["sppdev"].items()))
prefetch_hit_rates_formatted["vaampm"] = dict(sorted(prefetch_hit_rates_formatted["vaampm"].items()))


# bar plot with four groups of four bars (replacements first, then prefetchers)
fig, ax = plt.subplots()
bar_width = 0.15
opacity = 0.8
index = np.arange(len(lcc_hit_rates_formatted["no"].keys()))
ax.bar(index, lcc_hit_rates_formatted["no"].values(), bar_width, alpha=opacity, label='no', edgecolor='black')
ax.bar(index + bar_width, lcc_hit_rates_formatted["nextline"].values(), bar_width, alpha=opacity, label='next_line', edgecolor='black')
ax.bar(index + bar_width * 2, lcc_hit_rates_formatted["ipstride"].values(), bar_width, alpha=opacity, label='ip_stride', edgecolor='black')
ax.bar(index + bar_width * 3, lcc_hit_rates_formatted["sppdev"].values(), bar_width, alpha=opacity, label='spp_dev', edgecolor='black')
ax.bar(index + bar_width * 4, lcc_hit_rates_formatted["vaampm"].values(), bar_width, alpha=opacity, label='va_ampm_lite', edgecolor='black')
ax.set_xlabel('replacements')
ax.set_ylabel('LLC hit rate')
# ax.set_ylim(0, 1)
ax.set_title(file[5:-5])
ax.set_xticks(index + bar_width * 2)
ax.set_xticklabels(lcc_hit_rates_formatted["no"].keys())

# mark the highest combination
highest = max(max(lcc_hit_rates_formatted["no"].values()), 
              max(lcc_hit_rates_formatted["nextline"].values()), 
              max(lcc_hit_rates_formatted["ipstride"].values()), 
              max(lcc_hit_rates_formatted["sppdev"].values()), 
              max(lcc_hit_rates_formatted["vaampm"].values()))
for i, v in enumerate(lcc_hit_rates_formatted["no"].values()):
    if v == highest:
        ax.text(i - 0.09, v * 1.01, f'{v:.2f}', color='black', fontweight='bold')
for i, v in enumerate(lcc_hit_rates_formatted["nextline"].values()):
    if v == highest:
        ax.text(i + bar_width - 0.09, v * 1.01, f'{v:.2f}', color='black', fontweight='bold')
for i, v in enumerate(lcc_hit_rates_formatted["ipstride"].values()):
    if v == highest:
        ax.text(i + bar_width * 2 - 0.09, v * 1.01, f'{v:.2f}', color='black', fontweight='bold')
for i, v in enumerate(lcc_hit_rates_formatted["sppdev"].values()):
    if v == highest:
        ax.text(i + bar_width * 3 - 0.09, v * 1.01, f'{v:.2f}', color='black', fontweight='bold')
for i, v in enumerate(lcc_hit_rates_formatted["vaampm"].values()):
    if v == highest:
        ax.text(i + bar_width * 4 - 0.09, v * 1.01, f'{v:.2f}', color='black', fontweight='bold')


# scatter plot the prefetch hit rate on the same plot
ax2 = ax.twinx()
ax2.scatter(index, prefetch_hit_rates_formatted["no"].values(), marker='o', label='no', edgecolor='black')
ax2.scatter(index + bar_width, prefetch_hit_rates_formatted["nextline"].values(), marker='o', label='next_line', edgecolor='black')
ax2.scatter(index + bar_width * 2, prefetch_hit_rates_formatted["ipstride"].values(), marker='o', label='ip_stride', edgecolor='black')
ax2.scatter(index + bar_width * 3, prefetch_hit_rates_formatted["sppdev"].values(), marker='o', label='spp_dev', edgecolor='black')
ax2.scatter(index + bar_width * 4, prefetch_hit_rates_formatted["vaampm"].values(), marker='o', label='va_ampm_lite', edgecolor='black')
ax2.set_ylabel('prefetch hit rate')
ax2.set_ylim(0, 1)
# only set the ylim min to 0
# ax2.set_ylim(bottom=0)
# ax2.legend(loc='lower left', bbox_to_anchor=(0, 0))

# put legend outside
ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize='medium', framealpha=0.5)
ax2.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='medium', framealpha=0.5)
plt.tight_layout()
plt.savefig(f'plots/{file[5:-5]}_lcc_hit_rates.png', dpi=300)


# # bar plot with four groups of four bars (replacements first, then prefetchers)
# fig, ax = plt.subplots()
# bar_width = 0.15
# opacity = 0.8
# index = np.arange(len(prefetch_hit_rates_formatted["no"].keys()))
# ax.bar(index, prefetch_hit_rates_formatted["no"].values(), bar_width, alpha=opacity, label='no', edgecolor='black')
# ax.bar(index + bar_width, prefetch_hit_rates_formatted["nextline"].values(), bar_width, alpha=opacity, label='next_line', edgecolor='black')
# ax.bar(index + bar_width * 2, prefetch_hit_rates_formatted["ipstride"].values(), bar_width, alpha=opacity, label='ip_stride', edgecolor='black')
# ax.bar(index + bar_width * 3, prefetch_hit_rates_formatted["sppdev"].values(), bar_width, alpha=opacity, label='spp_dev', edgecolor='black')
# ax.bar(index + bar_width * 4, prefetch_hit_rates_formatted["vaampm"].values(), bar_width, alpha=opacity, label='va_ampm_lite', edgecolor='black')
# ax.set_xlabel('replacements')
# ax.set_ylabel('prefetch hit rate')
# ax.set_title(file[5:-5])
# ax.set_xticks(index + bar_width * 2)
# ax.set_xticklabels(prefetch_hit_rates_formatted["no"].keys())

# # put legend at bottom right
# ax.legend(loc='lower right', bbox_to_anchor=(1, 0))

# # mark the highest combination
# highest = max(max(prefetch_hit_rates_formatted["no"].values()), 
#               max(prefetch_hit_rates_formatted["nextline"].values()), 
#               max(prefetch_hit_rates_formatted["ipstride"].values()), 
#               max(prefetch_hit_rates_formatted["sppdev"].values()), 
#               max(prefetch_hit_rates_formatted["vaampm"].values()))
# for i, v in enumerate(prefetch_hit_rates_formatted["no"].values()):
#     if v == highest:
#         ax.text(i - 0.09, v * 1.01, f'{v:.2f}', color='black', fontweight='bold')
# for i, v in enumerate(prefetch_hit_rates_formatted["nextline"].values()):
#     if v == highest:
#         ax.text(i + bar_width - 0.09, v * 1.01, f'{v:.2f}', color='black', fontweight='bold')
# for i, v in enumerate(prefetch_hit_rates_formatted["ipstride"].values()):
#     if v == highest:
#         ax.text(i + bar_width * 2 - 0.09, v * 1.01, f'{v:.2f}', color='black', fontweight='bold')
# for i, v in enumerate(prefetch_hit_rates_formatted["sppdev"].values()):
#     if v == highest:
#         ax.text(i + bar_width * 3 - 0.09, v * 1.01, f'{v:.2f}', color='black', fontweight='bold')
# for i, v in enumerate(prefetch_hit_rates_formatted["vaampm"].values()):
#     if v == highest:
#         ax.text(i + bar_width * 4 - 0.09, v * 1.01, f'{v:.2f}', color='black', fontweight='bold')


# plt.tight_layout()
# plt.savefig(f'plots/{file[5:-5]}_prefetch_hit_rates.png', dpi=300)



