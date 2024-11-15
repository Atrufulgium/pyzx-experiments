import pyzx.extract as x
from pyzx.circuit.gates import CNOT
from pyzx.linalg import Mat2
from pyzx.utils import VertexType, EdgeType

from acircuit import *
from graph_helpers import *

from copy import deepcopy
from typing import Tuple

def extract_circuit_with_ancillae(
    g : Graph,
    optimize_czs : bool = True,
    optimize_cnots : int = 2,
    quiet : bool = True) -> ACircuit:
    """ pyzx' `extract_circuit`, but with ancillae if necessary.
    
    pyzx' `extract_circuit` relies on the presence of gflow, and gives up if
    it encounters the consequence of it not being present.

    This method instead adds ancilla(e) as necessary and continues extraction
    anyway. The resulting circuit will then require preperations and
    postselections, which may be acceptable.

    This does *not* modify the graph in place, and instead copies it, unlike
    pyzx' `extract_circuit`.
    """
    g = deepcopy(g)
    result : Circuit
    ancillae : List[Ancilla] = []
    while True:
        g_extract = deepcopy(g)
        try:
            result = x.extract_circuit(g_extract, optimize_czs, optimize_cnots, up_to_perm=False, quiet=quiet)
        except Exception as e:
            if len(e.args) != 1 or e.args[0] != "No extractable vertex found. Something went wrong":
                print(f"Something went wrong unexpectedly:\n{e}\nProblem graph:")
                zx.draw(g_extract, labels = True)
                raise e
            
            # TODO: Is "write to the original graph and restart" well-defined?
            # Does there exist an order of extraction that nullifies our effort?
            # Intuition says "no" but haven't proven it yet.
            turn_graph_extractable(g_extract, g, ancillae, quiet)
        else:
            break
    return ACircuit(result, ancillae)

def turn_graph_extractable(g_read : Graph, g_write : Graph, ancillae : List[Ancilla], quiet : bool):
    """ Adds an ancilla such that the extraction algorithm can find an extraction.
    
    This should be called when the frontier has no combination of rows that add
    up to a unit vector. In this case, we can still produce an unit vector at
    the cost of adding an ancilla connected to a rowsum except for one vertex.

    Here, `g_read` is the graph whose frontier biadjacency at the frontier is
    wrong, and `g_write` is the graph to write the resulting analysis' new
    spiders into. They may of course be the same.

    The `ancillae` list is added into with this calculation's result.
    """
    # The biadjacency matrix may be for instance (1 1 1 0 0).
    # In that case, this could add a row (0 1 1 0 0) to the biadj.
    # There's a lot to consider with this choice but at first I don't
    # care and want something that works.
    frontier = [list(g_read.neighbors(o))[0] for o in g_read.outputs()]
    neighbours = list(x.neighbors_of_frontier(g_read, frontier))
    num_cols = len(neighbours)
    m = x.bi_adj(g_read, neighbours, frontier)
    # Shitty choice I: just pick the first row.
    # Guaranteed to hold at least 2 non-zeroes.
    # (And as such also guaranteed to be a Mat2 and not Z2.)
    index : Tuple[int, slice] = (0, slice(num_cols))
    r = cast(Mat2, m[index])
    # Shitty choice II: make r's first 1 the extracted vert
    for i in range(num_cols):
        if r[0,i] == 1:
            r[0,i] = 0
            break
    # Guaranteed at least 1
    connect_with_ancilla : List[int] = []
    for i in range(num_cols):
        if r[0,i] == 1:
            connect_with_ancilla.append(neighbours[i])
            
    ancilla_qubit_circuit = len(frontier)
    ancilla_qubit_graph = ancilla_qubit_circuit
    row_input = g_read.row(g_read.inputs()[0])
    row_output = g_read.row(g_read.outputs()[0])
    ancillae.append(Ancilla(
        preparation = AncillaGate(VertexType.X, 0), # See _new_graph_to_swaps
        postselection=AncillaGate(VertexType.X, 0), # Instead of introducing a H-edge (v,o) below
        qubit=ancilla_qubit_circuit
    ))

    if not quiet:
        print("No extractable vertex found. Problem graph:")
        zx.draw(g_read, labels=True)
        print(f"Adding ancilla, connected to verts {connect_with_ancilla}.\nGraph before (prep/post not shown):")
        zx.draw(g_write, labels=True)
    # It would make sense to add the input as well and connect v to it.
    # However, pyzx turns it into a H-edge to have the nice double-
    # sided graph, which breaks this modification, and maintains
    # "no gflow".
    # Adding input is now the duty of _new_graph_to_swaps
    v = add_vertex(g_write, ancilla_qubit_graph, row_input + 1, VertexType.Z, phase = 0)
    o = add_vertex(g_write, ancilla_qubit_graph, row_output, VertexType.BOUNDARY, phase = None)
    g_write.add_edges([(v,o)], EdgeType.SIMPLE)
    g_write.add_edges([(v,w) for w in connect_with_ancilla], EdgeType.HADAMARD)
    g_write.set_outputs((*g_write.outputs(), o))
    if not quiet:
        print("Graph after (prep/post not shown):")
        zx.draw(g_write, labels=True)

_original_graph_to_swaps = x.graph_to_swaps
def _new_graph_to_swaps(g : Graph, no_swaps : bool = False) -> Circuit:
    """ Extens pyzx' `graph_to_swaps` to also handle our ancillae.
    
    This means that this supports the case where there are internal Z-spiders
    of 0 phase connected with with 1 input and multiple outputs.

    This type of graph is the result of calling `extract_circuit` on a graph
    with added additional outputs as in `extract_circuit_with_ancillae`.

    This method assumes that the ancillae are prepared into X-Phase 0.
    """
    # print("Graph before modifications")
    # zx.draw(g,labels = True)

    qubit_map : Dict[int,int] = { o: k for (k,o) in enumerate(g.outputs()) }
    # For every such "fork", we do the following:
    # - List the fork's prongs and delete the node itself;
    # - Choose one prong to be priviliged;
    # - Add a CNOT from the priviliged prong to every other*;
    # - Add a normal edge from the fork's former input to the priviliged prong;
    # - Add a normal edge from new inputs to all other prongs.
    # *Note that pyzx flipped the circuit up to this call, but once we arrive
    #  at this call, the circuit is chronological. These CNOTs must thus be
    #  *prepended* to the circuit.
    # However, we only have the pieces "swaps", "our stuff", "the rest".
    # ("The rest" is not accessible here.)
    # So we need to concat circuits.
    inserted_circuit = Circuit(len(g.outputs()))
    puts = set(g.inputs()).union(g.outputs())
    forks : List[int] = []
    for v in g.vertices():
        if v in puts:
            if len(g.neighbors(v)) > 1:
                raise ValueError(f"Input/output {v} is connected to multiple neighbours, which is unexpected.")
            continue
        
        connected_inputs = get_all_adjacent_where(g, v, lambda g,v,n: n in g.inputs())
        connected_internal = get_all_adjacent_where(g, v, lambda g,v,n : n not in puts)
        connected_outputs = get_all_adjacent_where(g, v, lambda g,v,n: n in g.outputs())
        if len(connected_internal) > 0 and len(connected_inputs) != 1:
            raise ValueError(f"Malformed fork {v}, which is unexpected; params {connected_inputs}, {connected_internal}, {connected_outputs}")
        if len(connected_outputs) > 1:
            forks.append(v)

    for fork in forks:
        input = get_all_adjacent_where(g, fork, lambda g,fork,n: n in g.inputs())[0]
        outputs = get_all_adjacent_where(g, fork, lambda g,fork,n: n in g.outputs())
        # Shitty choice III: Could take into account the cnot structure of the
        # rest of the resulting circuit.
        # Instead just grab the first.
        selected_output = outputs[0]
        del outputs[0]
        g.remove_vertex(fork)
        g.add_edge((input, selected_output), EdgeType.SIMPLE)
        for o in outputs:
            # Modify the graph
            i = add_vertex(g, g.num_inputs(), 0, VertexType.BOUNDARY, phase = None)
            g.add_edge((i,o), EdgeType.SIMPLE)
            g.set_inputs((*g.inputs(), i))
            # Modify the circuit
            inserted_circuit.add_gate(CNOT(qubit_map[selected_output], qubit_map[o]))
    
    # print("Updated graph:")
    # zx.draw(g, labels = True)
    # print("Inserted circuit:")
    # zx.draw(inserted_circuit)

    swaps = _original_graph_to_swaps(g, no_swaps)

    return swaps + inserted_circuit
x.graph_to_swaps = _new_graph_to_swaps