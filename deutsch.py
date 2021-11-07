###Deutsch's Algorithm
######################

from qiskit import QuantumCircuit, Aer

sim = Aer.get_backend('aer_simulator')

qc = QuantumCircuit(2, 1)

qc.x(1)
qc.barrier()

qc.h(0)

qc.h(1)

qc.barrier()

###constant oracle
##first bit is unchanged
##f(x) = 0
qc.i(1)


###balanced oracle
###CNOT, f(0) = 0, f(1) = 1
#qc.cx(0, 1)


qc.barrier()

qc.h(0)

qc.measure(0, 0)

print(qc)

result = sim.run(qc).result()
counts = result.get_counts()
print(counts)
##0 => function is constant
##1 => function is balanced