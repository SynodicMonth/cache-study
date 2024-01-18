from env import CacheEnv
import numpy as np
import torch
import torch.nn as nn

file = "data/llc_trace_403.txt"

env = CacheEnv(num_sets=2048, associativity=16, block_size=64)
env.load_trace(file)
print(f'loading trace file: {file}')
env.prepare_belady()
print('env ready')

observation, done = env.reset() # (associativity, 16)

preuses = []
ages = []

cnt = 0
while not done:
    set_index = env.cache.set_index(env.traces[env.curr_trace_idx][1])
    action = np.argmin(env.lru_last_use_cycles[set_index])
    next_observation, reward, done = env.step(action)
    observation = next_observation
    preuses.append(observation[1][1])
    ages.append(observation[1][3])

    if (env.curr_trace_idx > 500000):
        break
    cnt += 1
    if cnt % 10000 == 0:
        print(cnt)

print(len(preuses))

# plot the distribution of preuses and ages
import matplotlib.pyplot as plt
# print(preuses)

plt.hist(preuses, bins=100)
plt.savefig('preuse.png')
plt.clf()

plt.hist(ages, bins=100)
plt.savefig('age.png')
plt.clf()

# plot the distribution of preuses-ages
plt.hist(np.array(preuses) - np.array(ages), bins=100)
plt.savefig('preuse-age.png')