import pyzx as zx
from pyzx.graph.graph_s import GraphS as Graph
from pyzx.circuit import Circuit
from pyzx.circuit.gates import Gate
from pyzx.utils import FractionLike

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, cast
from typing_extensions import Self

_original_zx_draw = zx.draw
def _new_zx_draw(g, labels: bool = False, **kwargs):
    if isinstance(g, ACircuit):
        g = g.to_graph()
    _original_zx_draw(g, labels, **kwargs)
zx.draw = _new_zx_draw

@dataclass(frozen=True)
class AncillaGate:
    """ A gate that can be found in preperation or post-selection. """
    gate : Literal[1] | Literal[2] # VertexType.X, VertexType.Z, ugh
    phase : FractionLike

@dataclass(frozen=True)
class Ancilla:
    """ Denotes that a qubit is not a regular input/output, but ancillary. """
    preparation : AncillaGate
    postselection : AncillaGate
    qubit : int = -1

    def with_qubit(self, qubit : int) -> Self:
        return Ancilla(self.preparation, self.postselection, qubit)

class ACircuit:
    """ Represents a pyzx circuit that may have ancillae.
    
    Most methods here mirror pyzx' methods, see their documentation.

    This class *will* change the reference of `circuit` under your nose. Do not
    store it expecting it to stay the same, and instead store this ACircuit.
    In addition, passing it to most pyzx methods may work unexpectedly as they
    will not take into account ancillary data.
    """
    # Tried with inheritance; python's inheritance gave me a headache.

    __empty_qubit_circuit = Circuit(1, "1-qubit identity")

    def __init__(self, circuit : Circuit, ancillae : Iterable[Ancilla] = None):
        if circuit.bits > 0:
            raise NotImplementedError("No classical bits supported here.")

        self.circuit = circuit
        self.ancillae : List[Ancilla]
        if ancillae == None:
            self.ancillae = []
        else:
            self.ancillae : List[Ancilla] = [a for a in ancillae]

    def add_ancilla(self, ancilla : Ancilla, check_qubit = False) -> int:
        """ Adds an ancilla to the current circuit and returns its index. """
        target_qubit = self.circuit.qubits
        if check_qubit and ancilla.qubit != target_qubit:
            raise ValueError("The ancilla did not end up where it should.")
        ancilla = ancilla.with_qubit(target_qubit)
        self.ancillae.append(ancilla)
        self.circuit = self.circuit @ self.__empty_qubit_circuit
        return target_qubit
    
    def add_gate(self, gate : Gate):
        """ See pyzx' `Circuit.add_gate`, except for the kwargs part. """
        self.circuit.add_gate(gate)
    
    def prepend_gate(self, gate : Gate):
        """ See pyzx' `Circuit.prepend_gate`, except for the kwargs part. """
        self.circuit.prepend_gate(gate)
    
    def to_graph(self, zh = False, compress_rows = True, backend = None) -> Graph:
        """ Turns this circuit into a graph with its ancillae filled in.
        
        Do not use `ACircuit.circuit.to_graph()`, as it will not have the
        preperations or postselections.

        The arguments are the same as `Circuit.to_graph()`.
        """
        graph = self.circuit.to_graph(zh, compress_rows, backend)
        graph = cast(Graph, graph)

        inputs_remaining : Dict[int, Ancilla]  = { a.qubit: a for a in self.ancillae }
        ouptuts_remaining : Dict[int, Ancilla] = { a.qubit: a for a in self.ancillae }
        new_inputs : List[int] = []
        new_outputs : List[int] = []

        for v in graph.inputs():
            key = cast(int, graph.qubit(v))
            a = inputs_remaining.get(key)
            if a != None:
                graph.set_type(v, a.preparation.gate)
                graph.set_phase(v, a.preparation.phase)
                del inputs_remaining[key]
            else:
                new_inputs.append(v)
        
        if len(inputs_remaining) > 0:
            raise ValueError(f"There were preparations not mapped to graph inputs {graph.inputs()}:\n{inputs_remaining}")
        
        for v in graph.outputs():
            key = cast(int, graph.qubit(v))
            a = ouptuts_remaining.get(key)
            if a != None:
                graph.set_type(v, a.postselection.gate)
                graph.set_phase(v, a.postselection.phase)
                del ouptuts_remaining[key]
            else:
                new_outputs.append(v)
        
        if len(ouptuts_remaining) > 0:
            raise ValueError(f"There were postselections not mapped to graph outputs: {graph.outputs()}\n{ouptuts_remaining}")
        
        graph.set_inputs(new_inputs)
        graph.set_outputs(new_outputs)

        return graph
    
    def do_basic_optimization(self, do_swaps = True, quiet = True, to_basic_gates = True):
        """ Applies pyzx' `basic_optimization` to the circuit.

        This does not take into account ancillae. For instance, if a state is
        prepared into |0> and followed by an X gate, it won't be simplified
        into a |1>.
        """
        self.circuit = zx.optimize.basic_optimization(self.circuit.to_basic_gates(), do_swaps, quiet)
        if to_basic_gates:
            self.circuit = self.circuit.to_basic_gates()
    
    def compare_tensors(self, t2 : Graph | Circuit | Self, preserve_scalars = False):
        """ See pyzx' `compare_tensors`. """
        g1 = self.to_graph()
        g2 = graphify(t2)
        return zx.compare_tensors(g1, g2, preserve_scalars)
    
    def verify_equality(self, other : Graph | Circuit | Self, up_to_swaps = False) -> Literal[True] | None:
        """ See pyzx' `Circuit.verify_equality`. """
        other = cast(Graph, graphify(other).adjoint())
        this = self.to_graph()
        other = other + this
        # aaaaaa
        other = cast(Graph, other)
        zx.simplify.full_reduce(other)

        if not up_to_swaps:
            if other.is_id():
                return True
            return None
        n = self.circuit.qubits - len(self.ancillae)
        # Pretty sure zx throws if inputs are connected to inputs so this works
        foundputs = { v for e in other.edges() for v in e }
        if len(foundputs) == 2*n:
            return True
        return None
    
    def stats(self) -> str:
        return self.circuit.stats() + f"\n        {len(self.ancillae)} of the qubits is/are ancillary."
    
    def stats_dict(self) -> Dict[str, Any]:
        d = self.circuit.stats_dict()
        d["ancillae"] = len(self.ancillae)
        return d

def graphify(g : Graph | Circuit | ACircuit) -> Graph:
    """ Converts sort-of-graphs into definitely-graphs. """
    if isinstance(g, ACircuit):
        return g.to_graph()
    if isinstance(g, Circuit):
        return cast(Graph, g.to_graph())
    # pls
    return cast(Graph, g)