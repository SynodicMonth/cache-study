from nkache.model import QNetwork

import torch
import numpy as np
import csv

import matplotlib.pyplot as plt
import seaborn as sns


def calc_weights(model_path, observation_space, action_space, hidden_dim):
    model = QNetwork(observation_space, action_space, hidden_dim)
    model.load_state_dict(torch.load(model_path))

    weights = model.fc1.weight.data.numpy()
    weights = weights.mean(axis=0)

    assert len(weights) == observation_space

    line_weights = weights[9:].reshape(action_space, -1)
    line_weights = np.abs(line_weights).mean(axis=0)

    weights = np.concatenate((weights[:9], line_weights))

    abs_weights = np.abs(weights)
    min_weight = abs_weights.min()
    range_weight = abs_weights.max() - min_weight
    weights = (abs_weights - min_weight) / range_weight

    return weights

def calc_hitrate(stat_path):
    hit_rates = []

    with open(stat_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            accesses = int(row[3])
            miss = int(row[4])
            global_step = int(row[0])

            if accesses > 0:
                hit_rate = (accesses - miss) / accesses
                hit_rates.append((global_step, hit_rate))

    return hit_rates


AXIS_TEXT = [
    'address preuse',

    'load', 'rfo', 'write', 'prefetch', 'translation',

    'set index', 'set accesses', 'set accesses since miss',

    'line dirty', 'line preuse', 'age since insertion', 'age since last access',

    'last access load', 'last access rfo', 'last access write', 'last access prefetch', 'last access translation',
    
    'line load count', 'line rfo count', 'line write count', 'line prefetch count', 'line translation count',

    'hits since insertion', 'recency'
]

# MODEL_DIR = 'ckpts/403/'
EPISODE_START = 1
EPISODE_END = 60

MODELS = ['ckpts/403/policy_50.pt','ckpts/429/policy_50.pt','ckpts/437/policy_50.pt','ckpts/471/policy_50.pt','ckpts/483/policy_50.pt']

TRACES = ['403.gcc-48B', '429.mcf-217B', '437.leslie3d-273B', '471.omnetpp-188B', '483.xalancbmk-736B']

# for i in range(EPISODE_START, EPISODE_END + 1):
#     model_path = MODEL_DIR + f'policy_{i}.pt'
#     if os.path.exists(model_path):
#         MODELS.append(model_path)

def plot_heat_map():
    weight_mat = np.zeros((len(MODELS), len(AXIS_TEXT)))

    for i, model_path in enumerate(MODELS):
        weight_mat[i] = calc_weights(model_path, 265, 16, 128)
        
    plt.figure(figsize=(12, 5))
    sns.heatmap(weight_mat, cmap='Blues',
                xticklabels=AXIS_TEXT, yticklabels=TRACES)
    
    plt.tight_layout()
    plt.show()
    
def plot_stat_figure():
    plt.figure(figsize=(10, 10))
    for i in [403, 429, 437, 471, 483]:
        stat_path = f'stats/{i}/stat.csv'
        data = calc_hitrate(stat_path)
        
        steps, hit_rates = zip(*data)
        plt.plot(steps, hit_rates, label=str(i))
        
    plt.xlabel('Global Step')
    plt.ylabel('Hit Rate')
    plt.title('Hit Rate in Training')
    
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    plot_heat_map()
    plot_stat_figure()
