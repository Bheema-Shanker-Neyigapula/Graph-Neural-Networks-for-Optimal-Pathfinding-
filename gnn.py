import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt
import time
from networkx.algorithms.shortest_paths.weighted import dijkstra_path_length

# Define the Graph Neural Network (GNN) model
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, adjacency_matrix, feature_matrix):
        x = torch.mm(adjacency_matrix, feature_matrix)  # Aggregate neighbor features
        x = torch.relu(self.fc1(x))
        x = torch.mm(adjacency_matrix, x)  # Aggregate neighbor features again
        x = self.fc2(x)
        return x

# Define the optimal pathfinding function using GNN
def optimal_pathfinding(adjacency_matrix, feature_matrix, source_node, target_node):
    num_nodes = feature_matrix.shape[0]
    input_dim = feature_matrix.shape[1]
    hidden_dim = 16
    output_dim = 1

    model = GNN(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(100):
        optimizer.zero_grad()
        output = model(adjacency_matrix, feature_matrix)
        target = torch.zeros((num_nodes, output_dim))
        target[target_node] = 1  # Set target node as 1, rest as 0
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # Find the path with the highest GNN output from source to target
    current_node = source_node
    shortest_path = [current_node]
    max_iterations = num_nodes  # Set maximum number of iterations

    while current_node != target_node and max_iterations > 0:
        neighbors = torch.nonzero(adjacency_matrix[current_node]).squeeze()
        neighbor_scores = model(adjacency_matrix, feature_matrix)[neighbors]

        if len(neighbor_scores) == 0:
            break

        next_node = neighbors[torch.argmax(neighbor_scores)]
        current_node = next_node.item()
        shortest_path.append(current_node)
        max_iterations -= 1

    return shortest_path, model.fc1.weight.detach().numpy()

# Calculate accuracy of the shortest path
def calculate_accuracy(shortest_path, target_path, adjacency_matrix):
    intersection = set(shortest_path).intersection(target_path)
    accuracy = len(intersection) / len(target_path)
    dijkstra_length = dijkstra_shortest_path(adjacency_matrix, source_node, target_node)
    return accuracy, dijkstra_length

# Define a function for running Dijkstra algorithm
def dijkstra_shortest_path(adjacency_matrix, source_node, target_node):
    graph = nx.from_numpy_array(adjacency_matrix)
    length = dijkstra_path_length(graph, source=source_node, target=target_node)
    return length

# Example usage with larger data
adjacency_matrix = np.array([[0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                             [1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                             [0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
                             [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                             [0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
                             [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                             [0, 0, 0, 0, 0, 0, 1, 0, 1, 0]], dtype=np.float32)
feature_matrix = np.array([[0.2, 0.4],
                           [0.3, 0.6],
                           [0.4, 0.8],
                           [0.5, 1.0],
                           [0.1, 0.2],
                           [0.7, 0.9],
                           [0.3, 0.5],
                           [0.8, 0.2],
                           [0.6, 0.4],
                           [0.9, 0.7]], dtype=np.float32)
source_node = 0
target_node = 9

adjacency_tensor = torch.from_numpy(adjacency_matrix)
feature_tensor = torch.from_numpy(feature_matrix)

# Define the target path for accuracy calculation
target_path = [0, 1, 4, 5, 8, 9]

# Accuracy comparison
accuracy_values = []
dijkstra_lengths = []
try:
    for _ in range(100):
        shortest_path, _ = optimal_pathfinding(adjacency_tensor, feature_tensor, source_node, target_node)
        accuracy, dijkstra_length = calculate_accuracy(shortest_path, target_path, adjacency_matrix)
        accuracy_values.append(accuracy)
        dijkstra_lengths.append(dijkstra_length)

    # Plot the accuracy comparison graph
    plt.figure(1)
    plt.plot(accuracy_values, label='GNN')
    plt.plot(dijkstra_lengths, label='Dijkstra')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy / Shortest Path Length')
    plt.title('Accuracy and Shortest Path Length Comparison')
    plt.legend()

except Exception as e:
    print("An error occurred:", str(e))


# Efficiency comparison
execution_times = []
dijkstra_execution_times = []
try:
    for _ in range(100):
        shortest_path, _ = optimal_pathfinding(adjacency_tensor, feature_tensor, source_node, target_node)

        start_time = time.time()
        dijkstra_shortest_path(adjacency_matrix, source_node, target_node)
        dijkstra_execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        dijkstra_execution_times.append(dijkstra_execution_time)

        execution_time = 1000  # Placeholder for execution time calculation
        execution_times.append(execution_time)

    # Plot the efficiency comparison graph
    plt.figure(2)
    plt.plot(execution_times, label='GNN')
    plt.plot(dijkstra_execution_times, label='Dijkstra')
    plt.xlabel('Iterations')
    plt.ylabel('Execution Time (ms)')
    plt.title('Efficiency Comparison')
    plt.legend()

except Exception:
    print("An error occurred:", str(e))


# Scalability analysis
graph_sizes = [10, 20, 30, 40, 50]  # Vary the graph sizes as per your requirements
execution_times_scalability = []
dijkstra_execution_times_scalability = []

for size in graph_sizes:
    adjacency_matrix = np.ones((size, size), dtype=np.float32)
    feature_matrix = np.ones((size, 2), dtype=np.float32)
    adjacency_tensor = torch.from_numpy(adjacency_matrix)
    feature_tensor = torch.from_numpy(feature_matrix)

    start_time = time.time()
    shortest_path, _ = optimal_pathfinding(adjacency_tensor, feature_tensor, source_node, target_node)
    execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    execution_times_scalability.append(execution_time)

    start_time = time.time()
    dijkstra_shortest_path(adjacency_matrix, source_node, target_node)
    dijkstra_execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    dijkstra_execution_times_scalability.append(dijkstra_execution_time)

# Plot the scalability analysis graph
plt.figure(3)
plt.plot(graph_sizes, execution_times_scalability, marker='o', label='GNN')
plt.plot(graph_sizes, dijkstra_execution_times_scalability, marker='o', label='Dijkstra')
plt.xlabel('Graph Sizes')
plt.ylabel('Execution Time (ms)')
plt.title('Scalability Analysis')
plt.legend()


# Visualization of Learned Representations
_, learned_representations = optimal_pathfinding(adjacency_tensor, feature_tensor, source_node, target_node)

# Plot the visualization of learned representations
plt.figure(4)
plt.scatter(learned_representations[:, 0], learned_representations[:, 1])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Visualization of Learned Representations')

# Show all figures
plt.show()
