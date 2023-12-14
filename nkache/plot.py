import matplotlib.pyplot as plt
import numpy as np
import torch

# files
training_files = [
    "data/llc_trace_403_A.pt",
    "data/llc_trace_429_A.pt",
    "data/llc_trace_437_A.pt",
    "data/llc_trace_483_A.pt",
    "data/llc_trace_471_A.pt",
    # "data/llc_trace_astar_A.pt",
    "data/all_A.pt"
]

# labels of A parameters
labels = ['line dirty', 'line preuse', 'age since insertion', 'age since last access',
     'last access load', 'last access rfo', 'last access write', 'last access prefetch', 'last access translation',
     'line load count', 'line rfo count', 'line write count', 'line prefetch count', 'line translation count',
     'hits since insertion', 'recency']

# x axis: labels, y axis: A values for each file

# load A values
A_values = []
for file in training_files:
    A = torch.load(file).detach().numpy()
    # A = np.tanh(A)
    # minmax A
    A = A / np.max(np.abs(A))
    # A = np.tanh(A)
    A_values.append(A)
    print(A)


# Creating a heatmap with specified color mapping
fig, ax = plt.subplots()
im = ax.imshow(A_values, cmap='bwr')  # 'coolwarm' colormap for blue (low) to red (high)

# Setting axis ticks and labels
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(training_files)))
ax.set_xticklabels(labels)
ax.set_yticklabels(training_files)

# Rotating the x-axis labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Adding a title
ax.set_title("A values")

# Adding a colorbar
cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.set_ylabel("Values", rotation=-90, va="bottom")

# Adjust layout
fig.tight_layout()
plt.show()