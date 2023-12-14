import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from env import CacheEnv

# Suppose you have multiple training set files
training_file = 'data/llc_trace_437.pt'

# Assuming training_data is a list of (observation, action) pairs
observations = []
actions = []

training_data = torch.load(training_file)
for obs, action in training_data:
    observations.append(torch.tensor(obs, dtype=torch.float32))
    actions.append(torch.tensor(action, dtype=torch.float32))

# Convert lists to stacked tensors
observations = torch.stack(observations) # (batch_size, n_features)
actions = torch.stack(actions) # (batch_size, )

print(observations.shape)
print(actions.shape)

# Batch size
batch_size = 1000000

# Create a DataLoader for the training data
# dataset = TensorDataset(observations, actions)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the model parameters and optimizer
n_features = observations.shape[2]
A = torch.zeros(n_features)
A[15] = 1.0
A = torch.tensor(A, requires_grad=True)

optimizer = optim.SGD([A], lr=0.01)

losses = []
# Training loop
n_epochs = 3000
print_interval = 1
for epoch in range(n_epochs):
    optimizer.zero_grad()
    # Compute logits
    logits = torch.matmul(observations, A) # (batch_size, n_actions)

    # Apply softmax and compute loss
    # probabilities = nn.functional.softmax(logits, dim=1) # (batch_size, n_actions)
    # print(logits[2], actions[2])
    # loss = nn.functional.mse_loss(logits, actions)
    loss = nn.functional.cross_entropy(logits, torch.argmax(actions, dim=1))
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
torch.save(A, training_file.split(".")[0] + "_A.pt")

# plot losses

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

# env = CacheEnv(num_sets=2048, associativity=16, block_size=64)
# env.load_trace(training_file.split(".")[0] + ".txt")
# print(f'loading trace file: {training_file.split(".")[0] + ".txt"}')
# env.prepare_belady()
# print('env ready')
#
# # print('checking lru hit rate')
# # lru_hit_rate = env.lru_replacement_hit_rate()
# # print(f'lru hit rate: {lru_hit_rate}')
# # observation, done = env.reset()
#
# # print('checking optimal hit rate')
# # optimal_hit_rate = env.belady_replacement_hit_rate()
# # print(f'optimal hit rate: {optimal_hit_rate}')
# observation, done = env.reset()
#
# cnt = 0
# while not done:
#     # calc action (dot product)
#     logits = torch.matmul(torch.tensor(observation, dtype=torch.float32), A.detach())
#     # probabilities = nn.functional.softmax(logits, dim=0)
#     action = np.argmax(logits)
#     next_observation, reward, done = env.step(action)
#     observation = next_observation
#     cnt += 1
#     if cnt % 10000 == 0:
#         my_hit_rate = env.stats()['hit_rate']
#         print(f'inst: {env.curr_trace_idx}, hit rate: {my_hit_rate}')


