### Quantum Approximate Optimiziation Algorithm
###############################################

import matplotlib.pyplot as plt
import matplotlib.axes as axes
import numpy as np
import networkx as nx

from qiskit import Aer
from qiskit.tools.visualization import plot_histogram
from qiskit.circuit.library import TwoLocal
from qiskit_optimization.applications import Maxcut, Tsp
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

def draw_graph(G, colors, pos):
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)

def draw_tsp_solution(G, order, colors, pos):
    G2 = nx.DiGraph()
    G2.add_nodes_from(G)
    n = len(order)
    for i in range(n):
        j = (i + 1) % n
        G2.add_edge(order[i], order[j], weight=G[order[i]][order[j]]['weight'])
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G2, node_color=colors, edge_color='b', node_size=600, alpha=.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G2, 'weight')
    nx.draw_networkx_edge_labels(G2, pos, font_color='b', edge_labels=edge_labels)

# Generating a graph of 3 nodes

n = 4
num_qubits = n ** 2
tsp = Tsp.create_random_instance(n, seed=123)
adj_matrix = nx.to_numpy_matrix(tsp.graph)
print('distance\n', adj_matrix)

colors = ['r' for node in tsp.graph.nodes]
pos = [tsp.graph.nodes[node]['pos']  for node in tsp.graph.nodes]
draw_graph(tsp.graph, colors, pos)
plt.show()



###convert tsp to quadratic program
qp = tsp.to_quadratic_program()
qp2qubo = QuadraticProgramToQubo()
qubo = qp2qubo.convert(qp)

###convert to ising hamiltonian
qubitOp, offset = qubo.to_ising()


ee = NumPyMinimumEigensolver()
result = ee.compute_minimum_eigenvalue(qubitOp)

print('energy:', result.eigenvalue.real)
print('tsp objective:', result.eigenvalue.real + offset)
x = tsp.sample_most_likely(result.eigenstate)
print('feasible:', qubo.is_feasible(x))
z = tsp.interpret(x)
print('solution:', z)
print('solution objective:', tsp.tsp_value(z, adj_matrix))
draw_tsp_solution(tsp.graph, z, colors, pos)
plt.show()



# Solve using Quantum Computer (Simulation)


# algorithm_globals.random_seed = 123
# seed = 10598
# backend = Aer.get_backend('qasm_simulator')
# quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)
#
# spsa = SPSA(maxiter=300)
# ry = TwoLocal(qubitOp.num_qubits, 'ry', 'cz', reps=5, entanglement='linear')
# vqe = VQE(ry, optimizer=spsa, quantum_instance=quantum_instance)
#
# result = vqe.compute_minimum_eigenvalue(qubitOp)
#
# print('energy:', result.eigenvalue.real)
# print('time:', result.optimizer_time)
# x = tsp.sample_most_likely(result.eigenstate)
# print('feasible:', qubo.is_feasible(x))
# z = tsp.interpret(x)
# print('solution:', z)
# print('solution objective:', tsp.tsp_value(z, adj_matrix))
# draw_tsp_solution(tsp.graph, z, colors, pos)
# plt.show()