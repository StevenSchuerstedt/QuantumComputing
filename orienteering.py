import networkx as nx
import numpy as np
from docplex.mp.model import Model
import matplotlib.pyplot as plt
from qiskit.optimization.applications.ising.common import sample_most_likely
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
import itertools
from typing import Dict, List, Optional, Union
from qiskit_optimization.algorithms import OptimizationResult


# def interpret(
#         result: Union[OptimizationResult, np.ndarray]) -> List[Union[int, List[int]]]:
#     x = Union[OptimizationResult, np.ndarray]
#     if isinstance(result, OptimizationResult):
#         x = result.x
#     elif isinstance(result, np.ndarray):
#         x = result
#     else:
#         raise TypeError(
#             "Unsupported format of result. Provide an　OptimizationResult or a",
#             "binary array using np.ndarray instead of {}".format(type(result)),
#         )
#
#     n = int(np.sqrt(len(x)))
#     route = []  # type: List[Union[int, List[int]]]
#     for p__ in range(n):
#         p_step = []
#         for i in range(n):
#             if x[i * n + p__]:
#                 p_step.append(i)
#         if len(p_step) == 1:
#             route.extend(p_step)
#         else:
#             route.append(p_step)
#     return route

def draw_graph(G):
    labels = {n: str(n) + ';   ' + str(G.nodes[n]['weight']) for n in G.nodes}
    colors = [G.nodes[n]['weight'] for n in G.nodes]
    pos = nx.spring_layout(G)
    nx.draw(G, with_labels=True, labels=labels, node_color=colors)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
    plt.show()

def to_integer_program(G):
    mdl = Model(name="Orienteering")
    n = G.number_of_nodes()
    x = {
        (i, j): mdl.binary_var(name="x_{0}_{1}".format(i, j))
        for i in range(n)
        for j in range(n)
    }
    orienteering_func = mdl.sum(
        G.nodes[i]["weight"] * x[(i, j)]
        for i in range(n)
        for j in range(n)
    )
    mdl.maximize(orienteering_func)

    # Only 1 edge goes out from each node
    for i in range(n):
        mdl.add_constraint(mdl.sum(x[(i, j)] for j in range(n) if i != j) == 1)

    # Only 1 edge comes into each node
    for j in range(n):
        mdl.add_constraint(mdl.sum(x[(i, j)] for i in range(n) if i != j) == 1)

    # flow conservation conditions?
    # sackgassen vermeiden?
    for k in range(n):
        mdl.add_constraint(mdl.sum(x[(i, k)] for i in range(n)) <= 1)

    for k in range(n):
        mdl.add_constraint(mdl.sum(x[(k, i)] for i in range(n)) <= 1)

    mdl.add_constraint(mdl.sum(
                G.edges[i, j]["weight"] * x[(i, j)]
                for i in range(n)
                for j in range(n)
                if i != j
            ) <= T_max)

    # To eliminate sub-routes
    node_list = [i for i in range(n)]
    clique_set = []
    for i in range(2, len(node_list) + 1):
        for comb in itertools.combinations(node_list, i):
            clique_set.append(list(comb))
    for clique in clique_set:
        mdl.add_constraint(
            mdl.sum(x[(i, j)] for i in clique for j in clique if i != j) <= len(clique) - 1
        )

    op = from_docplex_mp(mdl)
    return op


# create a simple orienteering graph
G = nx.Graph()
G.add_node(0, weight=1)
G.add_node(1, weight=1)
G.add_node(2, weight=1)
#G.add_node(3, weight=2)
G.add_edge(0, 1, weight=10 )
G.add_edge(0, 2, weight=10 )
#G.add_edge(0, 3, weight=2 )
G.add_edge(1, 2, weight=10 )
#G.add_edge(1, 3, weight=5 )
#G.add_edge(2, 3, weight=7 )
T_max = 20
#draw_graph(G)


###convert tsp to quadratic program
qp = to_integer_program(G)
qp2qubo = QuadraticProgramToQubo()
qubo = qp2qubo.convert(qp)

###convert to ising hamiltonian
qubitOp, offset = qubo.to_ising()

#solve using numpy

ee = NumPyMinimumEigensolver()
result = ee.compute_minimum_eigenvalue(qubitOp)

print('energy:', result.eigenvalue.real)
print('tsp objective:', result.eigenvalue.real + offset)
#x = sample_most_likely(result.eigenstate)
#print('feasible:', qubo.is_feasible(x))
#z = interpret(x)
#print('solution:', z)
# print('solution objective:', tsp.tsp_value(z, adj_matrix))