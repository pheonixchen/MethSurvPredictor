import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
script_path=os.path.abspath(__file__)
script_dir=os.path.dirname(script_path)
os.chdir(script_dir)
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 100)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(100, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


input_dim = 51  
model = SimpleNN(input_dim)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()


weights = []
weights.append(model.fc1.weight.detach().numpy())
weights.append(model.fc2.weight.detach().numpy())
weights.append(model.fc3.weight.detach().numpy())


layers = [input_dim, 100, 100, 1]


def draw_neural_network(layers, weights):
    G = nx.Graph()

    
    pos = {}
    layer_nodes = []
    for i, num_nodes in enumerate(layers):
        layer_nodes.append([])
        for j in range(num_nodes):
            node_name = f'L{i}_N{j}'
            layer_nodes[-1].append(node_name)
            pos[node_name] = (i, -j + num_nodes // 2)  

    
    edges = []
    edge_colors = []
    for i in range(len(layers) - 1):
        for j, node in enumerate(layer_nodes[i]):
            for k, next_node in enumerate(layer_nodes[i+1]):
                weight = weights[i][k, j]
                edges.append((node, next_node))
                edge_colors.append(weight)

    G.add_edges_from(edges)

    
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=False, node_size=700, node_color='lightblue', 
            edge_color=edge_colors, edge_cmap=plt.cm.viridis, 
            width=2, edge_vmin=min(edge_colors), edge_vmax=max(edge_colors))

    
    for key, value in pos.items():
        plt.text(value[0], value[1] + 0.1, key, ha='center', va='center')

    plt.title("Neural Network Visualization")
    plt.show()

draw_neural_network(layers, weights)
