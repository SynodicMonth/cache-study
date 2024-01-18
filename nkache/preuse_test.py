import matplotlib.pyplot as plt
import numpy as np

BLOCKSIZE = 64

file = "data/llc_trace_437.txt"

with open(file, 'r') as f:
    lines = f.readlines()

last_last_use = {}
last_use = {}
preuses = []
ages = []
set_access_cnts = {}
accesses = []

t = 0
for line in lines:
    t += 1
    addr = line.split()[1]

    # get block address
    addr = int(addr, 16) // BLOCKSIZE

    # accesses.append(addr)

    set = addr % 2048
    if set not in set_access_cnts:
        set_access_cnts[set] = 0
    set_access_cnts[set] += 1

    if addr in last_last_use and addr in last_use:
        preuse = last_use[addr] - last_last_use[addr]
        age = set_access_cnts[set] - last_use[addr]
        preuses.append(preuse)
        ages.append(age)

    if addr in last_use:
        last_last_use[addr] = last_use[addr]
    last_use[addr] = set_access_cnts[set]

print(len(preuses))

# plt.plot(accesses[5000000:5000500])
# plt.savefig('accesses.png')
# plt.clf()

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
plt.rcParams['figure.figsize'] = (9, 3)
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

# plot the distribution of preuses and ages
plt.hist(preuses, bins=100)
plt.savefig('preuse.png')
plt.clf()

plt.hist(ages, bins=100)
plt.savefig('age.png')
plt.clf()

bias = np.array(preuses) - np.array(ages)

# calc the mean and std
mean = np.mean(bias)
std = np.std(bias)

print(f'mean: {mean}, std: {std}')

# plot the distribution of preuses-ages
plt.hist(bias, bins=100)
# plt.tight_layout()
plt.title(f'reuse-preuse (mean: {mean:.2f}, std: {std:.2f})')
plt.xlabel('reuse-preuse (set accesses between two uses)')
plt.ylabel('frequency')
plt.savefig('preuse-age.png')

    
