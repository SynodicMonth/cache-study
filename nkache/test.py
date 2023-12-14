from env import CacheEnv
import numpy as np
import torch
import torch.nn as nn

file = "data/llc_trace_437.txt"
# A = torch.load("data/all_A.pt").detach().numpy()
# knock to 0 if abs smaller than 0.3
# A[np.abs(A) < 0.5] = 0
# A = torch.tensor(A, dtype=torch.float32)
# print(A)

env = CacheEnv(num_sets=2048, associativity=16, block_size=64)
env.load_trace(file)
print(f'loading trace file: {file}')
env.prepare_belady()
print('env ready')

# print('checking lru hit rate')
# lru_hit_rate = env.lru_replacement_hit_rate()
# print(f'lru hit rate: {lru_hit_rate}')
# observation, done = env.reset()
#
# print('checking optimal hit rate')
# optimal_hit_rate = env.belady_replacement_hit_rate()
# print(f'optimal hit rate: {optimal_hit_rate}')
# observation, done = env.reset() # (associativity, 16)

# my_timestamps = []
# my_hitrate = []

# cnt = 0
# while not done:
#     # calc action (dot product)
#     logits = torch.matmul(torch.tensor(observation, dtype=torch.float32), A) # (associativity, 16) * (16, 1) = (associativity, 1)
#     # probabilities = nn.functional.softmax(logits, dim=0)
#     action = np.argmax(logits)
#     next_observation, reward, done = env.step(action)
#     observation = next_observation
#     my_timestamps.append(env.curr_trace_idx)
#     my_hitrate.append(env.stats()['hit_rate'])
#     if (env.curr_trace_idx > 300000):
#         break
#     cnt += 1
#     if cnt % 10000 == 0:
#         my_hit_rate = env.stats()['hit_rate']
#         print(f'inst: {env.curr_trace_idx}, hit rate: {my_hit_rate}')

observation, done = env.reset() # (associativity, 16)

lru_timestamps = []
lru_hitrate = []

# lru_A = np.zeros((16, 1))
# lru_A[15] = 1
# A = torch.tensor(lru_A, dtype=torch.float32)
cnt = 0
while not done:
    # calc action (dot product)
    # logits = torch.matmul(torch.tensor(observation, dtype=torch.float32), A) # (associativity, 16) * (16, 1) = (associativity, 1)
    # # probabilities = nn.functional.softmax(logits, dim=0)
    # action = np.argmax(logits)
    set_index = env.cache.set_index(env.traces[env.curr_trace_idx][1])
    action = np.argmin(env.lru_last_use_cycles[set_index])
    next_observation, reward, done = env.step(action)
    observation = next_observation
    lru_timestamps.append(env.curr_trace_idx)
    lru_hitrate.append(env.stats()['hit_rate'])
    if (env.curr_trace_idx > 300000):
        break
    cnt += 1
    if cnt % 10000 == 0:
        my_hit_rate = env.stats()['hit_rate']
        print(f'inst: {env.curr_trace_idx}, hit rate: {my_hit_rate}')

# observation, done = env.reset() # (associativity, 16)

# lfu_timestamps = []
# lfu_hitrate = []

# lfu_A = np.zeros((16, 1))
# lfu_A[2] = -1
# A = torch.tensor(lfu_A, dtype=torch.float32)
# cnt = 0
# while not done:
#     # calc action (dot product)
#     logits = torch.matmul(torch.tensor(observation, dtype=torch.float32), A) # (associativity, 16) * (16, 1) = (associativity, 1)
#     # probabilities = nn.functional.softmax(logits, dim=0)
#     action = np.argmax(logits)
#     next_observation, reward, done = env.step(action)
#     observation = next_observation
#     lfu_timestamps.append(env.curr_trace_idx)
#     lfu_hitrate.append(env.stats()['hit_rate'])
#     if (env.curr_trace_idx > 300000):
#         break
#     cnt += 1
#     if cnt % 10000 == 0:
#         my_hit_rate = env.stats()['hit_rate']
#         print(f'inst: {env.curr_trace_idx}, hit rate: {my_hit_rate}')

# observation, done = env.reset() # (associativity, 16)

# baledy_timestamps = []
# baledy_hitrate = []

# cnt = 0
# while not done:
#     action = np.argmax(env.belady_replacement_optimal(env.curr_trace_idx))
#     next_observation, reward, done = env.step(action)
#     observation = next_observation
#     baledy_timestamps.append(env.curr_trace_idx)
#     baledy_hitrate.append(env.stats()['hit_rate'])
#     if (env.curr_trace_idx > 300000):
#         break
#     cnt += 1
#     if cnt % 10000 == 0:
#         my_hit_rate = env.stats()['hit_rate']
#         print(f'inst: {env.curr_trace_idx}, hit rate: {my_hit_rate}')

# plot hit rate
import matplotlib.pyplot as plt
plt.plot(lru_timestamps, lru_hitrate, label='LRU')
# plt.plot(lfu_timestamps, lfu_hitrate, label='LFU')
# plt.plot(my_timestamps, my_hitrate, label='Mine')
# plt.plot(baledy_timestamps, baledy_hitrate, label='Belady')
plt.xlabel('timestamp')
plt.ylabel('hit rate')
plt.legend()
plt.savefig("a.png")


