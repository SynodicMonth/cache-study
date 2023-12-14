import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Suppose you have multiple training set files
training_files = ['data/llc_trace_403.pt', 'data/llc_trace_429.pt', 'data/llc_trace_437.pt', 'data/llc_trace_471.pt', 'data/llc_trace_483.pt']

all_observations_np = []
all_actions_np = []

# Load each training set and store data in NumPy arrays
for file in training_files:
    print(file)
    training_data = torch.load(file)
    observations_np = np.array([obs for obs, _ in training_data])
    actions_np = np.array([action for _, action in training_data])
    all_observations_np.append(observations_np)
    all_actions_np.append(actions_np)

# Concatenate all observations and actions using NumPy
all_observations_np = np.concatenate(all_observations_np, axis=0)
all_actions_np = np.concatenate(all_actions_np, axis=0)

# Convert back to PyTorch tensors
all_observations = torch.tensor(all_observations_np, dtype=torch.float32)
all_actions = torch.tensor(all_actions_np, dtype=torch.float32)


# Define the model parameters and optimizer
n_features = all_observations.shape[2]
A = torch.zeros(n_features)
A[15] = 1.0
A = torch.tensor(A, requires_grad=True)
optimizer = optim.Adam([A], lr=0.01)

# Training loop
n_epochs = 3000
print_interval = 1
losses = []
for epoch in range(n_epochs):
    optimizer.zero_grad()

    # Compute logits
    logits = torch.matmul(all_observations, A) # (batch_size, n_actions)

    # Apply softmax and compute loss
    loss = nn.functional.cross_entropy(logits, torch.argmax(all_actions, dim=1))
    losses.append(loss.item())

    # Backpropagation
    loss.backward()
    optimizer.step()

    if epoch % print_interval == 0:
        print(f"Epoch {epoch}, Average Loss: {loss.item()}")
        print(A)

# Print final value of A
print("Optimized A:", A)

# Save the model parameters
torch.save(A, "data/all_A.pt")

plt.figure(figsize=(12, 6))
plt.plot(losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Loss vs Epoch')
plt.show()

# plot a bar chart of A
feature_names = ['line dirty', 'line preuse', 'age since insertion', 'age since last access',
                 'last access load', 'last access rfo', 'last access write', 'last access prefetch', 'last access translation',
                 'line load count', 'line rfo count', 'line write count', 'line prefetch count', 'line translation count',
                 'hits since insertion', 'recency']

# Plotting
plt.figure(figsize=(12, 6))
plt.bar(feature_names, A.detach().numpy())
plt.xticks(rotation=45, ha="right")
plt.ylabel('Values')
plt.title('Feature Values')
plt.show()
