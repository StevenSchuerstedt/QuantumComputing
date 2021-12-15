---
title: Quantum Optimization
author: admin
type: page
date: 2021-11-08T16:36:38+00:00

---
$$\newcommand{\bra}[1]{\left<#1\right|}\newcommand{\ket}[1]{\left|#1\right>}\newcommand{\bk}[2]{\left<#1\middle|#2\right>}\newcommand{\bke}[3]{\left<#1\middle|#2\middle|#3\right>}$$

Using Quantum Computing for hard (np) combinatorial problems. These problems cannot really be solved efficiently classically (only by limiting to certain problem instances, using heuristics), so the question is if there could be a significant speedup using Quantum Computing.

### Quantum algorithm paradigms
- Quantum Fourier Transformation
- Grover Operator
- Harrow-Hassidim-Lloyd 
- variational quantum eingenvalue solver (VQE)
- direct Hamiltonian simulation (SIM)

### Traveling Salesman & Orienteering Problem
- NP Hard Problems (solution can be verified in polynomial time)
- can be stated as NP-Complete Problems (as hard as any problem in NP, only decision problems)

Note: only the worst case instances are NP hard (these instance are not easy to construct), in practice a lot of instance can be solved in P time. 

Combinatorial Optimization: discrete solutions with real value function
(e.g. Knapsack, Traveling Salesman)

Formulate these combinatorial Problems as linear programming problems. (linear programs)
in case of no real numbers allowed: integer programming

Ising Hamiltonian: MaxCut is equivalent to solving Ising model interpretation

IBM cplex (optimizer for classical computers)

- Qiskit Aqua provides many algorithms (e.g. VQE and also QUAO)

https://qiskit.org/documentation/apidoc/qiskit.aqua.algorithms.html

#### Heuristic Algorithms
- most powerful algorithms are heuristcs
=> sacrifice correctness, accuracy, precision for speed
(gradient descent, deep learning, genetic algorithm)
- in contrast to classical algorithm, that are theoretically perfect but slow

### Describe Combinatorial Problems as diagonal Hamiltonian (Ising Hamiltonian)

The Hamiltonian of any physical system describes its dynamics (movement). Naturally every system converges to its minimum energy state.

Generell Strategy: convert problem to hamiltonian (hermitian operator), solution to optimization problem is the largest (smallest) eigenvalue eigenvector.

Lets consider a boolen function `$f : \{ 0, 1 \} ^n \rightarrow \mathbb{R}$`, which takes a boolean string `$\textbf{x}$` of length n (the description of a specific instance of the combinatorial problem) and outputs a score, or some real number describing the length, time, points, edges etc. 

Now lets construct a diagonal hamiltonian such that,

$$H_f \ket{\textbf{x}} = f(\textbf{x}) \ket{\textbf{x}} $$

for `$H_f$` to represent `$f(\textbf{x})$`. The size of `$H_f$` grows exponentially with the number of qubits (like the vector representation of the register-bit ket). 

In a way `$H_f$` is a gigantic look up table, with all possible values of `$f(\textbf{x})$` on its diagonal and the solution to the optimization problem is the largest eigenvalue and corresponding eigenvector (note: eigenvectors of diagonal matrix are the cartesian basis vectors, eigenvalues are the values on the diagonal).

Constructing `$H_f$` explicitly is very inefficient (np-hard...) and does not give any computationally advantage, so we will not construct it explicitly, but give a description with its fourier expansion and Pauli-Z Operators. (pseudo boolean functions??)

$$Pauli_Z = Z = \begin{pmatrix} 1 & 0 \cr 0 & -1 \end{pmatrix}$$

##### Example: MaxCut
Lets start with the QUBO (?) (quadratic unconstrained binary optimization) formulation of the MaxCut Problem.

$$ C(\textbf{s}) = \frac{1}{2} \sum_{ij} (1-s_is_j)$$

Here `$s_i$` is a spin variable `$s_i \in \{ -1, 1 \} $`, but we can achieve a change of variables to binary with:

$$ C(\textbf{x}) = \frac{1}{2} \sum_{ij} (1-(-1)^{x_i}(-1)^{x_j})$$

or even

$$ C(\textbf{x}) = \sum_{ij} (x_i + x_j - 2x_ix_j)$$

Note: the sum goes only over all existing edge connections.

Now lets look at a specific instance of the MaxCut problem as an example.

{{< figure src="/maxcut.png" title="MaxCut" width="100%" >}}

The nodes can be in group 0 or 1. A bitstring like 0101 describes that the first node is in group 0, the second node in group 1 etc. (take care with ordering)

So to evaluate the cost function we can expand the sum to only account for the relevant (existing) edges (according to problem instance, e.g. picture).

$$C(\textbf{x}) = \frac{1}{2} ( (1-(-1)^{x_1}(-1)^{x_2}) + (1-(-1)^{x_1}(-1)^{x_4}) + (1-(-1)^{x_2}(-1)^{x_4}) + (1-(-1)^{x_2}(-1)^{x_3}) + (1-(-1)^{x_3}(-1)^{x_4}))$$

(five summands, as there are five edges)

Pluggin in values for `$\textbf{x}$` provides the according score, so in this case the number of edges being cut.

In order to construct the hamiltonian of this cost function in an effective way, we look at the action of the pauli-z operator acting on a register of qubits. Therefore we need to "make the pauli-z" bigger, to fit the dimension of the input vector, so we use the tensorproduct (kronecker product) with the identity matrix.

$$Z_i = I \otimes ... Z_i \otimes ... \otimes I $$

Where the Z gate appears at the ith position and only effects the ith qubit and there are as many identity matrizes as needed, to fit the size of the problem.

Similary 

$$Z_iZ_j = I \otimes ... Z_i \otimes ... Z_j \otimes...  \otimes I $$

Now the Z operator acting on a register of qubits gives

$$Z_i \ket{\textbf{x}} = (-1)^{x_i} \ket{\textbf{x}} $$

So for example 

$$IIZI \ket{0010} = -1 \ket{0010} $$

(because this is an eigenstate with eigenvalue -1)

The hamiltonian for the cost function is

$$C = \frac{1}{2} \sum_{ij} (I - Z_i Z_j) $$

In qiskit the sign is flipped and the constant terms (identity matrizes) are evaluated as an offset.

$$-C = \frac{1}{2} \sum_{ij} (-I + Z_i Z_j) $$

So four our five edge MaxCut example:

$$-C = \frac{1}{2} \sum_{ij}-I + \frac{1}{2}\sum_{ij}Z_i Z_j $$
$$-C = -2.5 I + \frac{1}{2}ZZII + \frac{1}{2}ZIIZ + \frac{1}{2}IZIZ + \frac{1}{2}IZZI + \frac{1}{2}IIZZ$$

The factors in front of the pauli-z operators are called fourier coefficients.
In general this representation can be achieved by using a fourier expansion on the original boolean function using multilinear polynomials, and then plugging in the fourier coefficients for the hamiltonian. (but calculating the fourier expansion is #P-hard)

$$ f(x) = \sum_{S \in [n]} \hat f(S)x^S $$

$$ C = \sum_{S \in [n]} \hat f(S) \prod_{j \in S} Z_j $$

=> cannot be efficiently computed (again np hard...) so use pseudo-boolean functions? (boolean function with boolean -> real ?)

multilinear polynomial representation (why is it called fourier?)



`$ x_S \in \{ -1, 1 \}$` or `$x_B \in \{ 0,1 \}$` with coordinate transform `$x_S = (-1)^{x_B}$`



$$H(\sigma) = \sum_{ij} J_{ij}\sigma_i \sigma_j$$

### VQE
Variational Quantum Eigensolver

=> find lowest eigenvalues of Hamiltonian (= Ising?)


### exponentiated matrix

- defined over taylor series
- exp(matrix) = some matrix
- rotation?
- exp von diagonal matrix
- i?

### QAOA

Quantum Approximate Optimization Algorithm

- Algorithm for combinatorial optimization problems (e.g. MaxCut or TSP)

- different paradigm than classical algorithms: only short computation on QC (because of errors), iterative with classical computer

- determine starting angles `$\gamma$` and `$\beta$` ? (=> done by the algorithm?)

- integer paramter p

- problem hamiltonian (as defined above), mixer hamiltonian

- exponentiated hamiltonian, exponentiated matrix ?

- QAOA needs Qubit Operator (Hamiltonian) this is implemented in Qiskit using adjacency matrix for graphs 

### Physik
- Hamiltonoperator operator für Energie
- => Observable für Energiewerte?
- Eigenwerte geben Werte für Energie an


