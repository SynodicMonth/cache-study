from nkache.model import QNetwork

import torch
import numpy as np
import os

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


AXIS_TEXT = [
    'address preuse',

    'load', 'rfo', 'write', 'prefetch', 'translation',

    'set index', 'set accesses', 'set accesses since miss',

    'line dirty', 'line preuse', 'age since insertion', 'age since last access',

    'last access load', 'last access rfo', 'last access write', 'last access prefetch', 'last access translation',
    
    'line load count', 'line rfo count', 'line write count', 'line prefetch count', 'line translation count',

    'hits since insertion', 'recency'
]

MODEL_DIR = 'ckpts/403/'
EPISODE_START = 1
EPISODE_END = 60

MODELS = []

for i in range(EPISODE_START, EPISODE_END + 1):
    model_path = MODEL_DIR + f'policy_{i}.pt'
    if os.path.exists(model_path):
        MODELS.append(model_path)

def plot_heat_map():
    weight_mat = np.zeros((len(MODELS), len(AXIS_TEXT)))

    for i, model_path in enumerate(MODELS):
        weight_mat[i] = calc_weights(model_path, 265, 16, 128)

    # tanspose
    weight_mat = weight_mat.T

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 10))
    sns.heatmap(weight_mat, cmap='Blues',
                xticklabels=MODELS, yticklabels=AXIS_TEXT)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_heat_map()
