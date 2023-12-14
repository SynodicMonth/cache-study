from env import CacheEnv
import numpy as np
import torch

filelist = ["data/llc_trace_437.txt",
            "data/llc_trace_483.txt",
            "data/llc_trace_astar.txt",
            "data/llc_trace_471.txt"]

# List to store the training data

for file in filelist:
    env = CacheEnv(num_sets=2048, associativity=16, block_size=64)
    training_data = []
    env.load_trace(file)
    print(f'loading trace file: {file}')
    env.prepare_belady()
    observation, done = env.reset()

    cnt = 0
    while not done:
        if len(training_data) > 300000:
            break
        action = env.belady_replacement_optimal(env.curr_trace_idx)

        next_observation, reward, done = env.step(np.argmax(action))

        if cnt % 5 == 0:
            action = np.nan_to_num(action)
            # too large, softmax will overflow
            action = action - np.max(action)
            # print(action)
            action = np.exp(action) / np.sum(np.exp(action))

            action = action.tolist()
            training_data.append((observation, action))

            if len(training_data) % 10000 == 0:
                print(len(training_data))

        observation = next_observation
        cnt += 1

    torch.save(training_data, file.split(".")[0] + ".pt")

# save training data
