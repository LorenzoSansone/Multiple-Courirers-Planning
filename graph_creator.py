# TODO: FIX THIS adding routes of couriers

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Input data as a string (for demonstration purposes)
data = """
3
7
15 10 7
3 2 6 8 5 4 4
0 3 3 6 5 6 6 2
3 0 4 3 4 7 7 3
3 4 0 7 6 3 5 3
6 3 7 0 3 6 6 4
5 4 6 3 0 3 3 3
6 7 3 6 3 0 2 4
6 7 5 6 3 2 0 4
2 3 3 4 3 4 4 0
"""

# Parse the input data
lines = data.strip().split('\n')
m = int(lines[0])
n = int(lines[1])
l = list(map(int, lines[2].split()))
s = list(map(int, lines[3].split()))
D = np.array([list(map(int, line.split())) for line in lines[4:]])
num_nodes = n + 1  # Total nodes including the origin
G = nx.Graph()
for i in range(num_nodes):  # 0-based indexing for nodes
    G.add_node(i)
for i in range(num_nodes):
    for j in range(i, num_nodes):
        if i != j:  # Avoid self-loops
            G.add_edge(i, j, weight=D[i][j])
pos = nx.spring_layout(G, weight='weight', scale=2.5)  # Increase scale for better spacing
plt.figure(figsize=(15, 12))  # Increase figure size for better visibility
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=700, font_size=12, font_weight='bold', edge_color='gray')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red')

plt.title('Graphical Representation of Distance Matrix')
plt.show()
