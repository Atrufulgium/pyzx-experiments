import pyzx as zx
from fractions import Fraction
from typing import Any, Dict, List, Set, Tuple, Union
from pyzx.circuit import Circuit, CNOT, HAD, ZPhase
from pyzx.graph.graph_s import GraphS as Graph
from pyzx.linalg import Mat2
from pyzx.utils import VertexType, EdgeType

import pyzx.extract as x

from graph_helpers import *

# Read from in `extract_inject`, written to nowhere except your usercode.
# Note that globals are only intuitive in a reference type, so doing it like this.
extra_params : Dict[str, Any] = {
    # A parameter determining performance.
    # Higher *generally* means better output, but runtime also scales O(q^max_greed).
    "max_greed": 4,
    # How to interpolate between prioritizing few edges (0) and small extract degree (1).
    "target_preference": 0.5
}

# (If only python had nicer reflection)
def extract_inject(g : Graph, frontier : List[int], circuit : Circuit, qubit_map : Dict[int,int]):
    """ This code is called before `extract_circuit` tries to extract any XY
    vert but after all frontier YZs are handled. Variables:
    - `g` (READ/WRITE): The partially extracted graph everything refers to.
    - `frontier` (READ): The frontier vertices in `g`.
      These vertices do not have phase, and you should maintain this when
      modifying `g`.
    - `circuit` (READ/WRITE): The circuit we're extracting to by *appending*.
    - `qubit_map` (READ): A current mapping of frontier vertices to circuit qubits.
    """
    max_greed = extra_params["max_greed"]
    α = extra_params["target_preference"]
    lcomp_optimise_frontier(g, frontier, max_greed, α, circuit, qubit_map)

def lcomp_optimise_frontier(g : Graph, frontier : List[int], max_greed : int, α : float, circuit : Circuit, qubit_map : Dict[int,int]):
    """ May reduce some edges near the frontier.
    
    Neighbours of the frontier may be interconnected. This is not relevant in
    the current extraction iteration, but will be in the next. In addition,
    the CNOTs do not affect that interconnection. So it may be worth it to
    reduce this interconnection.

    Local complementations preserve gflow, and on the frontier here, they flip
    precisely that interconnection.

    Unfortunately, it seems like "given `A` and `b`, find `x` with minimal
    `Hamming(Ax, b)`" is pretty hard. So here we're brute forcing it for (which
    works for small qubit counts) with a given cap `max_greed`.

    This has complexity `O(N^2 q^q')` with `N` the spider count, `q` the
    output qubit count, and `q' = min(q,max_greed)`.
    (With infinite max_greed it's actually `O(N^2 2^q)` instead but who does
    that.)

    This method targets only the vertices `(v) -- (f) -- output` and not multi-
    connected frontier vertices `=(f) -- output` for minimalisation.
    """
    if max_greed <= 0:
        return

    # Update the frontier in advance to not incorporate any vertices not
    # incident to good frontier neighbours.
    # A good neighbour is one that's about to be extracted by just being
    # a single edge.
    good_frontier_vertices : Set[int] = set()
    for v in frontier:
        if len(g.neighbors(v)) == 2:
            good_frontier_vertices.add(v)
    
    good_neighbours : Set[int] = set()
    for v in frontier:
        neighs = get_all_adjacent_where(g, v, lambda g,v,n: g.type(n) != VertexType.BOUNDARY)
        if len(neighs) == 1:
            # These frontier vertices are of the form `(w) -- (v) -- output`.
            # Grab the non-BOUNDARY vert.
            good_neighbours.add(neighs[0])
    
    if len(good_neighbours) == 0:
        raise ValueError("Did not call this method from the right place -- only call this from that specific spot in apply_cnots.")
    
    possibly_relevant_verts : Set[int] = set()
    for w in good_neighbours:
        possibly_relevant_verts.update(g.neighbors(w))
    
    # We will be considering everything in `frontier` from here on out.
    # The only stuff that makes sense is what influences `good_neighbours`.
    # These are precisely the frontier vertices in `possibly_relevant_verts`.
    # As `good_frontier_vertices` have exactly one neighbour, they can be
    # excluded.
    frontier = [f for f in frontier if f in possibly_relevant_verts and not f in good_frontier_vertices]
    
    # What connections the neighbourhood of the frontier starts out with.
    # Convention: all edges are stored as (smaller index, larger index).
    initial_config : Set[Tuple[int,int]] = set()
    # The score is only influenced by vertices targeted by extraction-
    # incident edges. After all, we want to minimize the connections that get
    # moved to the frontier, not take into account unrelated verts.
    initial_config_scored : Set[Tuple[int,int]] = set()
    # In the same order as `frontier`, what edges are flipped by lcomping.
    # We select at most `min(len(frontier), max_greed)` of these.
    primitives : List[Set[Tuple[int,int]]] = []
    primitives_scored : List[Set[Tuple[int,int]]] = []
    # Also needed in the lcomps later.
    neighbourhoods : List[Set[int]] = []

    frontier_neighbourhood : Set[int] = set()
    frontier_boundaries : List[int] = []
    for v in frontier:
        boundary_neighbourhood = [ u for u in g.neighbors(v) if g.type(u) == VertexType.BOUNDARY ]
        if len(boundary_neighbourhood) != 1:
            raise ValueError("Given frontier vert is not connected to exactly 1 boundary vert!")
        frontier_boundaries.append(boundary_neighbourhood[0])

        neighbourhood = set(g.neighbors(v)).difference(boundary_neighbourhood)
        frontier_neighbourhood.update(g.neighbors(v))
        complete_neighbourhood = {(i,j) for i in neighbourhood for j in neighbourhood if i < j}
        scored_neighbourhood = {e for e in complete_neighbourhood if e[0] in good_neighbours or e[1] in good_neighbours}
        primitives.append(complete_neighbourhood)
        primitives_scored.append(scored_neighbourhood)
        neighbourhoods.append(neighbourhood)
    
    for v in frontier_neighbourhood:
        for u in frontier_neighbourhood:
            if u >= v:
                continue
            if g.connected(u,v):
                initial_config.add((u,v))
    
    initial_config_scored = {e for e in initial_config if e[0] in good_neighbours or e[1] in good_neighbours}
    
    # The best configuration of primitives so far.
    # It stores the indices of the entries in the `primitives` list we want to
    # lcomp in the graph.
    # (The initial state isn't even invalid, it represents the default!)
    best : Set[int] = set()
    best_score = α * len(initial_config_scored) + (1-α) * len(initial_config)

    actual_greed = min(max_greed, len(frontier))
    for k in range(actual_greed):
        for selected in n_choose_k_up_to_order(len(frontier), k):
            current_config = set(initial_config)
            current_config_scored = set(initial_config_scored)
            for i in selected:
                current_config.symmetric_difference_update(primitives[i])
                current_config_scored.symmetric_difference_update(primitives_scored[i])
            #print(f"Config {selected}: {current_config}")
            current_score = α * len(current_config_scored) + (1-α) * len(current_config)
            if current_score < best_score:
                best_score = current_score
                best = set(selected)
    
    # Actually apply the lcomps we calculated. This needs to:
    # - Change the neighbourhoods (the whole thing we care about)
    # - Add π/2 to the neighbours
    # - Add a HAD ZPHASE(-π/2) HAD to the circuit
    for i in best:
        for e in primitives[i]:
            g.add_edge_smart(e, EdgeType.HADAMARD)
        for v in neighbourhoods[i]:
            g.set_phase(v, g.phase(v) + Fraction(1,2))
        
        circ_qubit = qubit_map[frontier[i]]
        circuit.add_gate(HAD(circ_qubit))
        circuit.add_gate(ZPhase(circ_qubit, Fraction(-1,2)))
        circuit.add_gate(HAD(circ_qubit))

        boundary_edge = (frontier[i], frontier_boundaries[i])
        if  g.edge_type(boundary_edge) == EdgeType.HADAMARD:
            raise ValueError("This should've been extracted already by cleanup")










# The below code is copied directly from pyzx/extract.py with the following modifications:
# - The types BaseGraph, VT, ET are replaced with Graph (GraphS), int, Tuple[int,int] resp.
#   (Only because my linter's crap and can't handle the generics somewhy.)
# - All extract.py-local methods are now prefixed with `x.` as I've imported
#   pyzx.extract like that.
# - Any block of # ADDED CODE .. # END ADDED CODE
def extract_circuit(
        g: Graph,
        optimize_czs: bool = True,
        optimize_cnots: int = 2,
        up_to_perm: bool = False,
        quiet: bool = True
        ) -> Circuit:
    """Given a graph put into semi-normal form by :func:`~pyzx.simplify.full_reduce`, 
    it extracts its equivalent set of gates into an instance of :class:`~pyzx.circuit.Circuit`.
    This function implements a more optimized version of the algorithm described in
    `There and back again: A circuit extraction tale <https://arxiv.org/abs/2003.01664>`_

    Args:
        g: The ZX-diagram graph to be extracted into a Circuit.
        optimize_czs: Whether to try to optimize the CZ-subcircuits by exploiting overlap between the CZ gates
        optimize_cnots: (0,1,2,3) Level of CNOT optimization to apply.
        up_to_perm: If true, returns a circuit that is equivalent to the given graph up to a permutation of the inputs.
        quiet: Whether to print detailed output of the extraction process.

    Warning:
        Note that this function changes the graph `g` in place. 
        In particular, if the extraction fails, the modified `g` shows 
        how far the extraction got. If you want to keep the original `g`
        then input `g.copy()` into `extract_circuit`.
    """

    gadgets = {}
    inputs = g.inputs()
    outputs = g.outputs()

    c = Circuit(len(outputs))

    for v in g.vertices():
        if g.vertex_degree(v) == 1 and v not in inputs and v not in outputs:
            n = list(g.neighbors(v))[0]
            gadgets[n] = v

    qubit_map: Dict[int,int] = dict()
    frontier = []
    for i, o in enumerate(outputs):
        v = list(g.neighbors(o))[0]
        if v in inputs:
            continue
        frontier.append(v)
        qubit_map[v] = i

    czs_saved = 0
    q: Union[float, int]
    
    while True:
        # ADDED CODE (my way too unsophisticated linter is yelling at me)
        greedy : List[CNOT] = []
        cnots : List[CNOT] = []
        # END ADDED CODE

        # preprocessing
        czs_saved += x.clean_frontier(g, c, frontier, qubit_map, optimize_czs)
        
        # Now we can proceed with the actual extraction
        # First make sure that frontier is connected in correct way to inputs
        neighbor_set = x.neighbors_of_frontier(g, frontier)
        
        if not frontier:
            break  # No more vertices to be processed. We are done.
        
        # First we check if there is a phase gadget in the way
        if x.remove_gadget(g, frontier, qubit_map, neighbor_set, gadgets):
            # There was a gadget in the way. Go back to the top
            continue
            
        neighbors = list(neighbor_set)
        m = x.bi_adj(g, neighbors, frontier)
        if all(sum(row) != 1 for row in m.data):  # No easy vertex

            if optimize_cnots > 1:
                greedy_operations = x.greedy_reduction(m)
            else:
                greedy_operations = None

            if greedy_operations is not None:
                greedy = [CNOT(target, control) for control, target in greedy_operations]
                if (len(greedy) == 1 or optimize_cnots < 3) and not quiet:
                    print("Found greedy reduction with", len(greedy), "CNOT")
                cnots = greedy

            if greedy_operations is None or (optimize_cnots == 3 and len(greedy) > 1):
                perm = x.column_optimal_swap(m)
                perm = {v: k for k, v in perm.items()}
                neighbors2 = [neighbors[perm[i]] for i in range(len(neighbors))]
                m2 = x.bi_adj(g, neighbors2, frontier)
                if optimize_cnots > 0:
                    cnots = m2.to_cnots(optimize=True)
                else:
                    cnots = m2.to_cnots(optimize=False)
                # Since the matrix is not square, the algorithm sometimes introduces duplicates
                cnots = x.filter_duplicate_cnots(cnots)

                if greedy_operations is not None:
                    m3 = m2.copy()
                    for cnot in cnots:
                        m3.row_add(cnot.target,cnot.control)
                    reductions = sum(1 for row in m3.data if sum(row) == 1)
                    if greedy and (len(cnots)/reductions > len(greedy)-0.1):
                        if not quiet: print("Found greedy reduction with", len(greedy), "CNOTs")
                        cnots = greedy
                    else:
                        neighbors = neighbors2
                        m = m2
                        if not quiet: print("Gaussian elimination with", len(cnots), "CNOTs")
            # We now have a set of CNOTs that suffice to extract at least one vertex.
        else:
            if not quiet: print("Simple vertex")
            cnots = []

        # ADDED CODE
        #extracted = x.apply_cnots(g, c, frontier, qubit_map, cnots, m, neighbors)
        extracted = apply_cnots(g, c, frontier, qubit_map, cnots, m, neighbors)
        # END ADDED CODE
        if not quiet: print("Vertices extracted:", extracted)
            
    if optimize_czs:
        if not quiet: print("CZ gates saved:", czs_saved)
    # Outside of loop. Finish up the permutation
    x.id_simp(g, quiet=True)  # Now the graph should only contain inputs and outputs
    # Since we were extracting from right to left, we reverse the order of the gates
    c.gates = list(reversed(c.gates))
    return x.graph_to_swaps(g, up_to_perm) + c

def apply_cnots(g: Graph, c: Circuit, frontier: List[int], qubit_map: Dict[int, int],
                cnots: List[CNOT], m: Mat2, neighbors: List[int]) -> int:
    """Adds the list of CNOTs to the circuit, modifying the graph, frontier, and qubit map as needed.
    Returns the number of vertices that end up being extracted"""
    if len(cnots) > 0:
        cnots2 = cnots
        cnots = []
        for cnot in cnots2:
            m.row_add(cnot.target, cnot.control)
            cnots.append(CNOT(qubit_map[frontier[cnot.control]], qubit_map[frontier[cnot.target]]))
        x.connectivity_from_biadj(g, m, neighbors, frontier)

    # ADDED CODE
    extract_inject(g, frontier, c, qubit_map)
    # The only thing this updates is interconnection between one of the classes
    # the biadjacency matrix cares about.
    # As such, it does *not* need to be recomputed.
    # END ADDED CODE

    good_verts = dict()
    for i, row in enumerate(m.data):
        if sum(row) == 1:
            v = frontier[i]
            w = neighbors[[j for j in range(len(row)) if row[j]][0]]
            good_verts[v] = w
    if not good_verts:
        raise Exception("No extractable vertex found. Something went wrong")
    hads = []
    outputs = g.outputs()
    for v, w in good_verts.items():  # Update frontier vertices
        hads.append(qubit_map[v])
        # c.add_gate("HAD",qubit_map[v])
        qubit_map[w] = qubit_map[v]
        b = [o for o in g.neighbors(v) if o in outputs][0]
        g.remove_vertex(v)
        g.add_edge(g.edge(w, b))
        frontier.remove(v)
        frontier.append(w)

    for cnot in cnots:
        c.add_gate(cnot)
    for h in hads:
        c.add_gate("HAD", h)

    return len(good_verts)