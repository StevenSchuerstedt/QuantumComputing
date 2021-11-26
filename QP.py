from typing import Dict, List, Optional, Union

import networkx as nx
import numpy as np
from docplex.mp.model import Model
from qiskit_optimization.applications import Maxcut, Tsp
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_node(0, weight=8)
G.add_node(1, weight=5)
G.add_node(2, weight=3)
G.add_edge(0, 1, weight=4.7 )
G.add_edge(0, 2, weight=4.7 )
G.add_edge(1, 2, weight=4.7 )

labels = {n: G.nodes[n]['weight'] for n in G.nodes}
colors = [G.nodes[n]['weight'] for n in G.nodes]
pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, labels=labels, node_color=colors)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
plt.show()

print(G.nodes[0]['weight'])


tsp = Tsp.create_random_instance(3, seed=123)



def draw_graph(G, colors, pos):
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)


mdl = Model(name="TSP")
n = tsp._graph.number_of_nodes()


colors = ['r' for node in tsp.graph.nodes]
pos = [tsp.graph.nodes[node]['pos']  for node in tsp.graph.nodes]
draw_graph(tsp.graph, colors, pos)
#plt.show()

#(0,0) (0,1) (0,2)
#(1,0) (1,1) (1,2)
#(2,0) (2,1) (2,2)
#nodes und verbindungen? wurde ein node besucht?

x = {
    (i, k): mdl.binary_var(name="x_{0}_{1}".format(i, k))
    for i in range(n)
    for k in range(n)
}

#mdl.print_information()
#mdl.prettyprint()
#print(x[(0, 0)])

#print(tsp._graph.edges[0, 2]["weight"]* x[(0, 1)])

tsp_func = mdl.sum(
    #kantengewicht der kante die nodes i, j verbindet
    tsp._graph.edges[i, j]["weight"] * x[(i, k)] * x[(j, (k + 1) % n)]
    for i in range(n)
    for j in range(n)
    for k in range(n)
    if i != j
)
mdl.minimize(tsp_func)
for i in range(n):
    mdl.add_constraint(mdl.sum(x[(i, k)] for k in range(n)) == 1)
for k in range(n):
    mdl.add_constraint(mdl.sum(x[(i, k)] for i in range(n)) == 1)


#op = from_docplex_mp(mdl)