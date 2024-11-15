from fractions import Fraction
import pyzx as zx
from pyzx.graph.graph_s import GraphS as Graph
from pyzx.utils import VertexType, EdgeType, FractionLike
import random
from typing import List, Set
from typing_extensions import Self

from graph_helpers import *

PhaseGadgetID = int
PhaseGadgetSet = Set[PhaseGadgetID]

class PhaseGadgetForm:
    """ Class that describes a ZX-diagram in reduced phase gadget form.

    A diagram in phase-gadget form is a diagram on `n` qubits with `m` phase
    gadgets where each qubit can be connected with some of the gadgets.

    See also theorem 4.21 in There and Back Again.
    """

    def __init__(self, qubits : int, phase_gadgets : int, random_seed : int = 0):
        """ Creates a new reduced phase gadget form graph. 
        
        This graph has a fixed number of qubits and phase gadgets.
        
        It is initialised with all phase gadgets having a phase of π/4 and no
        connections to the phase gadgets."""
        self.qubits = qubits
        # Also implicitely stores gadget count in the length
        self.phase_gadget_phases : List[FractionLike] = [Fraction(1,4) for _ in range(phase_gadgets)]
        # Each index represents what gadgets that qubit is connected with
        self.connectivity : List[PhaseGadgetSet] = [set() for _ in range(qubits)]
        self.rng = random.Random(random_seed)
        # Each index represents what qubit its input is copied from
        self.copied_from : List[int | None] = [None for _ in range(qubits)]
    
    def set_copies(self, qubit : int, target_qubits : Set[int]) -> Self:
        """ Makes certain qubits a copy of another qubit.

        Tells the circuit that certain qubits do not have free input, but
        instead start with a |0〉 state and get their input copied from `qubit`.

        The given set `target_qubits` must have even size and not overlap with
        the target qubits of another qubit.
        """
        if len(target_qubits) % 2 != 0:
            raise ValueError(f"The given set {target_qubits} of target qubits must be even.")
        if qubit in target_qubits:
            raise ValueError(f"The given set {target_qubits} contains the qubit to copy from, {qubit}, itself. This is not allowed.")

        old_target_qubits = set()
        other_target_qubits = set()
        for q in range(self.qubits):
            copy_from = self.copied_from[q]
            if copy_from == None:
                continue
            if copy_from == qubit:
                old_target_qubits.add(qubit)
            else:
                other_target_qubits.add(qubit)

        intersection = other_target_qubits.intersection(target_qubits)
        if len(intersection) != 0:
            raise ValueError(f"Set {target_qubits} overlaps with another set (at indices {intersection}).")
        
        for q in old_target_qubits:
            self.copied_from[q] = None
        for q in target_qubits:
            self.copied_from[q] = qubit

        return self
    
    def set_phase(self, gadget : PhaseGadgetID, phase : FractionLike):
        """ Sets the phase of a given gadget. 
        
        As usual, `phase` is [0,2) representing [0,2π) as in the rest of pyzx.
        """
        self.phase_gadget_phases[gadget] = phase
    
    def get_phase(self, gadget : PhaseGadgetID) -> FractionLike:
        """ Gets the phase of a given gadget.
        
        As usual, `phase` is [0,2) representing [0,2π) as in the rest of pyzx.
        """
        return self.phase_gadget_phases[gadget]
    
    def set_many_phases(self, phases : List[FractionLike]) -> Self:
        given_phases = len(phases)
        actual_phases = len(self.phase_gadget_phases)
        if given_phases != actual_phases:
            raise ValueError(f"The amount of given phases ({given_phases}) does not equal the number of phase gadgets of this diagram ({actual_phases})")
        
        self.phase_gadget_phases = phases
        return self
    
    def set_connection(self, qubit : int, gadget : PhaseGadgetID, active : bool = True) -> Self:
        """ Sets whether a qubit is connected to a gadget.
        
        This method returns itself for easy chaining.
        """
        max_index = len(self.phase_gadget_phases)
        if gadget >= max_index:
            raise ValueError(f"Gadget {gadget} does not exist (outside 0 up to {max_index - 1}).")

        if active:
            self.connectivity[qubit].add(gadget)
        else:
            self.connectivity[qubit].difference({gadget})
        return self
    
    def get_connection(self, qubit : int, gadget : PhaseGadgetID) -> bool:
        """ Returns whether a qubit is connected to a gadget. """
        max_index = len(self.phase_gadget_phases)
        if gadget >= max_index:
            raise ValueError(f"Gadget {gadget} does not exist (outside 0 up to {max_index - 1}).")
        
        return gadget in self.connectivity[qubit]

    
    def set_many_connections(self, qubit : int, connected_gadgets : PhaseGadgetSet) -> Self:
        """ Removes all connections of a qubit and replaces them with the given connections.
        
        This method returns itself for easy chaining.
        """
        max_index = len(self.phase_gadget_phases)
        for gadget in connected_gadgets:
            if gadget >= max_index:
                raise ValueError(f"Phase gadget set {connected_gadgets} contains gadgets that don't exist (outside 0 up to {max_index - 1}).")

        self.connectivity[qubit] = connected_gadgets
        return self
    
    def set_many_connections_random(self, qubit : int, probability : float = 0.5) -> Self:
        """ Coinflips all connections between a qubit and all gadget. """
        gadget_count = len(self.phase_gadget_phases)
        gadget_set = {i for i in range(gadget_count) if self.rng.random() < probability}
        return self.set_many_connections(qubit, gadget_set)
    
    def set_all_connections_random(self, probability : float = 0.5) -> Self:
        """ Coinflips all connections between all qubits and gadgets. """
        for q in range(self.qubits):
            self.set_many_connections_random(q, probability)
        return self
    
    def to_graph(self, do_copies : bool = False) -> Graph:
        """ Turns this into an actual graph pyzx can work with."""
        # Following structure (coordinates (q,r)):
        # Input Z spiders at (2,1), (3,1), ...
        # Phase gadgets at ((0,2),(1,2)), ((0,3),(1,3)), ...
        # Output Z spiders at (2,2+#gadgets), (3, 2+#gadgets), ...
        # And of course the pyzx boundary spiders.
        g = Graph()

        pyzx_inputs : List[int] = []
        graph_inputs : List[int] = []
        graph_outputs : List[int] = []
        pyzx_outputs : List[int] = []
        gadget_bottoms : List[int] = []
        gadget_phases : List[int] = []

        qubit_count = self.qubits
        gadget_count = len(self.phase_gadget_phases)

        for q in range(qubit_count):
            pyzx_inputs.append(add_vertex(g, qubit=2+q, row=0, vertextype=VertexType.BOUNDARY, phase=None))
            graph_inputs.append(add_vertex(g, qubit=2+q, row=1, vertextype=VertexType.Z, phase=0))
            graph_outputs.append(add_vertex(g, qubit=2+q, row=2+gadget_count, vertextype=VertexType.Z, phase=0))
            pyzx_outputs.append(add_vertex(g, qubit=2+q, row=3+gadget_count, vertextype=VertexType.BOUNDARY, phase=None))
        
        for gadget in range(gadget_count):
            phase = self.phase_gadget_phases[gadget]
            gadget_bottoms.append(add_vertex(g, qubit=1, row=2+gadget, vertextype=VertexType.Z, phase=0))
            gadget_phases.append(add_vertex(g, qubit=0, row=2+gadget, vertextype=VertexType.Z, phase=phase))
        
        # Adding the edges now is easy
        # Horizontal
        g.add_edges([(pyzx_inputs[q], graph_inputs[q]) for q in range(qubit_count)], EdgeType.SIMPLE)
        g.add_edges([(graph_inputs[q], graph_outputs[q]) for q in range(qubit_count)], EdgeType.HADAMARD)
        g.add_edges([(graph_outputs[q], pyzx_outputs[q]) for q in range(qubit_count)], EdgeType.SIMPLE)

        # Vertical
        g.add_edges([(gadget_bottoms[gadget], gadget_phases[gadget]) for gadget in range(gadget_count)], EdgeType.HADAMARD)

        # Funky
        for q in range(qubit_count):
            input = graph_inputs[q]
            gadgets = self.connectivity[q]
            g.add_edges([(input, gadget_bottoms[gadget]) for gadget in gadgets], EdgeType.HADAMARD)

        # If we include the copies in this graph already, add those CNOTs and
        # remove those qubits from the inputs/outputs.
        if do_copies:
            pyzx_inputs_set = set(pyzx_inputs)
            pyzx_outputs_set = set(pyzx_outputs)
            remove = set()

            for q in range(qubit_count):
                copy_from = self.copied_from[q]
                if copy_from == None:
                    continue
                inp = pyzx_inputs[q]
                out = pyzx_outputs[q]
                g.add_edges([(inp, graph_inputs[copy_from]), (out, graph_outputs[copy_from])], EdgeType.SIMPLE)
                g.set_type(inp, VertexType.X)
                g.set_type(out, VertexType.X)
                remove.add(inp)
                remove.add(out)
            
            pyzx_inputs_set.difference_update(remove)
            pyzx_outputs_set.difference_update(remove)
            pyzx_inputs = list(pyzx_inputs_set)
            pyzx_outputs = list(pyzx_outputs_set)
        
        g.set_inputs(pyzx_inputs)
        g.set_outputs(pyzx_outputs)
        
        return g
    
    def to_extracted_circ(self, silent : bool = True) -> zx.Circuit:
        # Do the copies manually afterwards as otherqise pyzx is going to
        # compile into the smaller circuit we explicitely do not want.
        g = self.to_graph(do_copies=False)
        if not silent:
            print("Generated graph (without copies):")
            zx.draw(g)

        g.normalize()
        circ = zx.extract_circuit(g)

        # Now add the copy data.
        for q in range(self.qubits):
            copy_from = self.copied_from[q]
            if copy_from == None:
                continue
            circ.prepend_gate(zx.gates.CNOT(control=copy_from, target=q))
            circ.add_gate(zx.gates.CNOT(control=copy_from, target=q))

        circ = zx.optimize.basic_optimization(circ.to_basic_gates(), do_swaps=True).to_basic_gates()
        
        if not silent:
            print("Extracted circuit:")
            zx.draw(circ)
            print(circ.stats(depth=True))
        
        return circ