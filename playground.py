from fractions import Fraction
import os
from random import random
import pyzx as zx
from pyzx import Circuit
from pyzx.graph.graph_s import GraphS as Graph
from typing import Tuple, List, cast, Literal
from extract_helpers import *

zx.settings.drawing_backend = 'matplotlib'

fname = os.path.join('./mod5_4_before.txt')
circ = zx.Circuit.load(fname).to_basic_gates()

g = cast(Graph, circ.to_graph())
zx.draw(g)

from pyzx.utils import VertexType
def vertTypeToString(vt: int):
    match vt:
        case VertexType.BOUNDARY: return "BOUNDARY"
        case VertexType.Z: return "Z"
        case VertexType.X: return "X"
        case VertexType.H_BOX: return "H_BOX"
        case _: return "Unknown"

for v in g.vertices():
    print(f"Vertex {v}:")
    print(f"  Neighbours: {[n for n in g.neighbors(v)]}")
    print(f"  Phase: {g.phase(v)}")
    print(f"  In?: {v in g.inputs()} Out?: {v in g.outputs()}")
    print(f"  Type: {vertTypeToString(g.type(v))}")
    out : List[Measurement] = []
    if (try_get_measurement(g, v, out)):
        print(f"  Measurement: {out[0]}")

g3 = Graph()
g3.add_vertices(5)
g3.set_inputs([0])
g3.set_outputs([4])
for v in [1,2,3]:
    g3.set_type(v, VertexType.Z)
g3.add_edges([(0,2),(1,2),(3,2),(4,2)], EdgeType.SIMPLE)
g3.set_phase(1, Fraction(1,2))
g3.set_phase(3, Fraction(1,3))
i : int = 0
for v in [(0,0), (1,1), (0,2), (1,3), (0,4)]:
    g3.set_position(i, v[0], v[1])
    i += 1


from pyzx.linalg import Mat2, MatLike
m = Mat2([[1,1,1],[0,1,1], [1,1,0]])
print(m)
x : MatLike = []
y : MatLike = []
m.gauss(x=x, y=y)

g5 = Graph()
for (p1,p2) in [(0,0), (0,1), (1,1)]:
    g4 = Graph()
    g4.add_vertices(21)
    for v in range(2,17):
        g4.set_position(v, 2, v)
    for v in {0,1,17,18}:
        g4.set_position(v, 3, v)
    g4.set_position(19, 4.5, 6.5)
    g4.set_position(20, 4.5, 11.5)
    g4.set_phase(19, p1)
    g4.set_phase(20, p2)
    for v in {1,17,19,20}.union(range(2,17)):
        g4.set_type(v, VertexType.Z)
    g4.add_edges([(0,1), (17,18)], EdgeType.SIMPLE)
    g4.add_edges([(19,20)], EdgeType.HADAMARD)
    g4.add_edges([(1,v) for v in range(2,18)], EdgeType.HADAMARD)
    g4.add_edges([(19,v) for v in range(2,12)], EdgeType.HADAMARD)
    g4.add_edges([(20,v) for v in range(7,17)], EdgeType.HADAMARD)
    v = 2
    for _ in range(3):
        g4.set_phase(v, Fraction(v,23))
        v += 1
        [u1,u2] = list(g4.add_vertices(2))
        g4.set_position(u1, 1, v)
        g4.set_type(u1, VertexType.Z)
        g4.set_phase(u1, Fraction(1,2))
        g4.set_position(u2, 0, v)
        g4.set_type(u2, VertexType.X)
        g4.set_phase(u2, Fraction(v,23))
        g4.add_edges([(u2,u1), (u1,v)], EdgeType.SIMPLE)
        v += 1
        [u1] = list(g4.add_vertices(1))
        g4.set_position(u1, 1, v)
        g4.set_type(u1, VertexType.X)
        g4.set_phase(u1, Fraction(v,23))
        g4.add_edges([(u1,v)], EdgeType.SIMPLE)
        v += 1
        [u1] = list(g4.add_vertices(1))
        g4.set_position(u1, 1, v)
        g4.set_type(u1, VertexType.Z)
        g4.set_phase(u1, Fraction(v,23))
        g4.add_edges([(u1,v)], EdgeType.HADAMARD)
        v += 1
        [u1] = list(g4.add_vertices(1))
        g4.set_position(u1, 1, v)
        g4.add_edges([(u1,v)], EdgeType.SIMPLE)
        v += 1
    g5 = g5.tensor(g4)

g6 = Graph()
g6.add_vertices(11)
for v in range(2,8):
    g6.set_position(v, 2, v)
for v in {0,1,8,9}:
    g6.set_position(v, 3, v)
g6.set_position(10, 4.5, 4)
g6.set_phase(10, Fraction(1,2))
for v in {10}.union(range(1,9)):
    g6.set_type(v, VertexType.Z)
g6.add_edges([(0,1), (8,9)], EdgeType.SIMPLE)
g6.add_edges([(1,v) for v in range(2,9)], EdgeType.HADAMARD)
g6.add_edges([(10,v) for v in range(2,8)], EdgeType.HADAMARD)
v = 2
g6.set_phase(v, Fraction(v,23))
v += 1
[u1] = list(g6.add_vertices(1))
g6.set_position(u1, 1, v)
g6.set_type(u1, VertexType.Z)
g6.set_phase(u1, Fraction(v,23))
g6.add_edges([(u1,v)], EdgeType.SIMPLE)
v += 1
[u1,u2] = list(g6.add_vertices(2))
g6.set_position(u1, 1, v)
g6.set_type(u1, VertexType.Z)
g6.set_phase(u1, Fraction(1,2))
g6.set_position(u2, 0, v)
g6.set_type(u2, VertexType.X)
g6.set_phase(u2, Fraction(v,23))
g6.add_edges([(u2,u1), (u1,v)], EdgeType.SIMPLE)
v += 1
[u1] = list(g6.add_vertices(1))
g6.set_position(u1, 1, v)
g6.set_type(u1, VertexType.X)
g6.set_phase(u1, Fraction(v,23))
g6.add_edges([(u1,v)], EdgeType.SIMPLE)
v += 1
[u1] = list(g6.add_vertices(1))
g6.set_position(u1, 1, v)
g6.set_type(u1, VertexType.Z)
g6.set_phase(u1, Fraction(v,23))
g6.add_edges([(u1,v)], EdgeType.HADAMARD)
v += 1
[u1] = list(g6.add_vertices(1))
g6.set_position(u1, 1, v)
g6.add_edges([(u1,v)], EdgeType.SIMPLE)

s = {24, 27, 28, 31}
for v in s:
    g6.set_type(s, VertexType.X)
g6.set_inputs(set(g6.inputs()).difference(s))
g6.set_outputs(set(g6.outputs()).difference(s))
g6.add_edges([(24,21),(28,21),(27,22),(28,22)], EdgeType.SIMPLE)

import pyzx as zx
from pyzx import Circuit
circ_pre = Circuit(3)
circ_pre.add_gate(zx.gates.CNOT(2,1))

circ_post = Circuit(3)
circ_post.add_gate(zx.gates.ZPhase(1, Fraction(1,4)))
circ_post.add_gate(zx.gates.CNOT(1,2))

circ = zx.generate.CNOT_HAD_PHASE_circuit(10, 100)

import random as rng
from extract_ancilla import *

# Completely Bernoulli(0.5) random phaseless graph
import random as rng
rng.seed(2)

size : int = 2
internal : int = 4
g7 = Graph()
# Boundary inputs, boundary outputs, inputs, outputs
verts = g7.add_vertices(size * 4)
for v in verts:
    if v < size * 2:
        g7.set_type(v, VertexType.BOUNDARY)
    else:
        g7.set_type(v, VertexType.Z)
    qubit = v % size
    row : FloatInt
    if v < size:
        row = -1
    elif v < 2 * size:
        row = size + 1
    elif v < 3 * size:
        row = 0
    else:
        row = size
    g7.set_position(v, qubit, row)
# The rest
verts = g7.add_vertices(internal)
for v in verts:
    g7.set_type(v, VertexType.Z)
    qubit = rng.random() * (size - 1)
    row = rng.random() * size
    g7.set_position(v, qubit, row)
g7.set_inputs([i for i in range(size)])
g7.set_outputs([i + size for i in range(size)])
g7.add_edges((i, i + size * 2) for i in range (size * 2))
max = size * 4 + internal
add_edges = []
for u in range(size * 2, max):
    for v in range(u, max):
        if rng.random() > 0.5:
            add_edges.append((u,v))
g7.add_edges(add_edges, EdgeType.HADAMARD)
zx.draw(g7, labels=True)

ac = extract_circuit_with_ancillae(deepcopy(g7), quiet = False)
zx.draw(g7)
zx.draw(ac)
print(ac.compare_tensors(g7))
print(ac.stats())

g3 = Graph()
add_vertex(g3, 0, 0, VertexType.BOUNDARY, None)
add_vertex(g3, 1, 0, VertexType.BOUNDARY, None)
add_vertex(g3, 0, 2, VertexType.BOUNDARY, None)
add_vertex(g3, 1, 2, VertexType.BOUNDARY, None)
add_vertex(g3, 0.5, 1, VertexType.Z, Fraction(1,4))
g5.set_inputs([0,1])
g5.set_outputs([2,3])
g5.add_edges([(i, 4) for i in range(4)], EdgeType.HADAMARD)