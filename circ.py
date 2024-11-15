import pyzx as zx
from pyzx import Circuit
from pyzx.circuit import ZPhase, XPhase, HAD
from pyzx.graph.graph_s import GraphS as Graph
from typing import Tuple, List, cast

from extract_helpers import *

class PartiallyExtracted():
    """ A container for work in progress extraction. """
    def __init__(self, unextracted: Graph):
        unextracted = deepcopy(unextracted)
        og = OpenGraph(unextracted)
        self.open_graph = og
        self.circuit = Circuit(len(og.outputs))
        # ZX diagrams seem to allow anything as qubit index.
        # Circuits just seem like Range(0,n)
        # This maps qubit -> circuit index
        self.qubit_mapping = { unextracted.qubit(og.outputs[i]): i for i in range(len(og.outputs))}

    def cleanup_outputs(self) -> bool:
        """ Removes all LCs at and CZs between outputs.
        
        Returns whether there were any modifications to the graph.
        
        This just applies the following three methods until nothing changes:
        - `extract_output_local_cliffords()`
        - `extract_fused_outputs()`
        - `extract_arity_2_outputs()`
        """
        updates = False
        while True:
            updated = self.extract_output_local_cliffords()
            updated |= self.extract_fused_outputs()
            updated |= self.extract_arity_2_outputs()
            updates |= updated
            if not updated:
                return updates

    def cnot_outputs(self, src, dst):
        """ Given a source and destination output vertex, adds a CNOT after. 
        
        This affects the graph by having `src`'s neighbourhood symdiff'd with
        `dist`'s.
        """
        if src == dst:
            raise ValueError(f"You can't CNOT with the same source and target {src}, silly.")
        og = self.open_graph
        g = og.original_graph

        if not src in og.outputs or not dst in og.outputs:
            raise ValueError("Malformed graph; src or dst aren't an output.")
        
        # Neighbours in the open graph to not read the BOUNDARY vertices.
        src_neighbours = set(og.graph.neighbors(src))
        dst_neighbours = set(og.graph.neighbors(dst))
        
        # src's neighbours are now symmetrically diff'd with dst's
        g.remove_edges([(src, n) for n in src_neighbours])
        g.add_edges([(src, n) for n in src_neighbours.symmetric_difference(dst_neighbours)], EdgeType.HADAMARD)

        # add the cnot
        self.circuit.prepend_gate(zx.gates.CNOT(self.get_circuit_qubit(src), self.get_circuit_qubit(dst)))

        self.open_graph = OpenGraph(g)
    
    def extract_output_local_cliffords(self) -> bool:
        """ Extracts any local cliffords adjacent to the frontier.
        
        These LCs are the output wires that may have been removed when
        murdering pyzx' BOUNDARY verts.

        These are self-reported by OpenGraph's constructor.
        """
        og = self.open_graph
        g = og.original_graph
        updates = False
        for qubit in og.output_lcs.keys():
            circuit_qubit = self.get_circuit_qubit_from_qubit(qubit)
            # These arity-2 stuffs leave an output endpoint and a graph endpoint.
            # To connect them, keep track of everything and remove everything
            # that gets removed.
            edge_verts : Set[int] = set()
            locals = og.output_lcs[qubit]
            if len(locals) == 0:
                continue

            to_remove : Set[int] = set()
            for (vert_type, vert_phase, v) in locals:
                # H-boxes are encoded on the edges and not as a vertex and
                # require special handling.
                [n] = g.neighbors(v)
                if vert_type == VertexType.H_BOX:
                    g.set_edge_type((v,n), EdgeType.SIMPLE)
                    self.circuit.prepend_gate(HAD(circuit_qubit))
                    updates = True
                else:
                    edge_verts.update(g.neighbors(v))
                    match vert_type:
                        case VertexType.BOUNDARY: raise ValueError("Uhhhh there's like six things that should've gone wrong to trigger this im not even gonna bother writing a proper error")
                        case VertexType.Z: self.circuit.prepend_gate(ZPhase(circuit_qubit, vert_phase))
                        case VertexType.X: self.circuit.prepend_gate(XPhase(circuit_qubit, vert_phase))
                    edge_verts.difference({v})
                    to_remove.add(v)
                    updates = True
            if len(edge_verts) == 0: # Only a had
                continue
            # any other case now has 2
            if len(edge_verts) != 2:
                raise ValueError(f"Another impossible branch: {edge_verts}")
            g.remove_vertices(to_remove)
            [e1, e2] = edge_verts
            g.add_edges([(e1, e2)], EdgeType.SIMPLE)
        self.open_graph = OpenGraph(g)
        return updates
    
    def extract_fused_outputs(self) -> bool:
        """ Unfuses connected frontier vertices and extracts their gates.
        
        Returns whether there were any modifications to the graph.
        """
        og = self.open_graph
        g = og.original_graph
        updates = False
        for u in og.outputs:
            for v in og.outputs:
                if u == v:
                    continue
                if g.connected(u,v):
                    if not g.edge_type((u,v)) == EdgeType.HADAMARD:
                        raise ValueError("Non-Hadamard connection between output vertices!")
                    qu = self.get_circuit_qubit(u)
                    qv = self.get_circuit_qubit(v)
                    g.remove_edge((u,v))
                    self.circuit.prepend_gate("CZ", qu, qv)
                    updates = True
        self.open_graph = OpenGraph(g)
        return updates
    
    def extract_arity_2_outputs(self):
        """ Extracts output vertices that only connect to one internal vertex.
        This turns their neighbour into the new frontier, and grabs its XY
        measurement as gate.

        Returns whether there were any modifications to the graph.
        """
        og = self.open_graph
        g = og.original_graph
        updates = False
        for u in og.outputs:
            nu = list(g.neighbors(u))
            if len(nu) != 2:
                continue
            if u in og.phases and og.phases[u] != 0:
                raise ValueError(f"Output vertex {u} is measured {og.planes[u]}({og.phases[u]}), which is not allowed.")
            if g.type(nu[0]) != VertexType.BOUNDARY != g.type(nu[1]):
                raise ValueError(f"Vertex {u} is somehow not an output despite OpenGraph claiming it is.")
            v = nu[0] if g.type(nu[0]) == VertexType.Z else nu[1]
            o = nu[0] if v == nu[1] else nu[1]
            # The following is not an error and could actually happen.
            if v in og.planes and og.planes[v] != "XY":
                continue

            # Don't do anything if we touch an input.
            if v in og.inputs:
                continue

            # We finally have =(v:α)--(u:0)-|- where we want to extract to
            #   =(v:0)-|-(α)-□-
            # (Where | represents the extraction boundary.)
            qubit = self.get_circuit_qubit(u)
            self.circuit.prepend_gate("H", qubit)
            self.circuit.prepend_gate("ZPhase", qubit, phase=og.phases[v])
            # Note that pyzx's diagrams have an additional boundary after what
            # I call the boundary, which is the `o` here. This also needs to be
            # updated.
            g.remove_vertex(o)
            g.set_type(u, VertexType.BOUNDARY)
            g.set_edge_type((u,v), EdgeType.SIMPLE)
            replace_measurement(g, v, "XY", 0, og.original_graph_measurement_artifacts[v])
            g.set_qubit(v, g.qubit(u))
            new_outputs = set(g.outputs())
            # no need to remove o from outputs as it's already remove_vertex'd
            new_outputs.add(u)
            g.set_outputs(new_outputs)
            updates = True
        self.open_graph = OpenGraph(g)
        return updates

    def extract_swaps_and_input_LCs(self):
        """ Handles the final swaps and input LCs (Step 5 in T&BA).

        Once the graph consists of just inputs connected to outputs wiht some
        LCs on the input side, this implements the swap gates needed and puts
        the final LCs afterwards.
        """
        if len(self.open_graph.inputs) != len(self.open_graph.outputs):
            raise ValueError("This map is not unitary and cannot be extracted to a circuit.")
        perm = [-1 for _ in self.open_graph.inputs]

        raise NotImplementedError()

    def get_circuit_qubit(self, v : int) -> int:
        """ Given vertex `v` of the unextracted graph, returns what circuit
        qubit this lies on. Not *that* well-defined, but it tries.
        """
        # First try to see if this neighbours a BOUNDARY in the original graph.
        # If not, try to read the dictionary directly; it may not exist then.
        
        og = self.open_graph
        g = og.original_graph
        candidates : List[int] = []
        for w in g.neighbors(v):
            if g.type(w) == VertexType.BOUNDARY:
                candidates.append(w)
        
        if len(candidates) == 1:
            return self.get_circuit_qubit_from_qubit(g.qubit(candidates[0]))

        return self.get_circuit_qubit_from_qubit(self.open_graph.original_graph.qubit(v))
    
    def get_circuit_qubit_from_qubit(self, q : FloatInt) -> int:
        """ Given qubit `q` of the unextracted graph, returns what circuit
        qubit this lies on. Not *that* well-defined, but it tries.
        """
        return self.qubit_mapping[q]
    
    def draw(self):
        print("Unextracted:")
        zx.draw(self.open_graph.original_graph, labels=True)
        print("Extracted:")
        zx.draw(self.circuit)



# [v] Step -1: Circuit => MBQC+LC
#      → Circuit.to_graph()? Moet output nog nagaan.
# [v] Step  0: MBQC+LC => Phase Gadgdet Form (Prop 4.16)
# [v]     Step 0.1: Itereer diagram totdat niet meer beschikbaar:
# [~]         Step 0.1.1: Als u~v in YZ, pivot uv (Lem 2.32) en MBWC'fy (Lem 4.5)
#              → rewrite rule API pivot
# [~]         Step 0.1.2: Als een u in XZ, local compl u (Lem 2.31) en MBWC'fy (Lem 4.3)
#              → rewrite rule API lcomp
# (≈) Step  1: Remove LC en unfuse frontier (triv).
# [v] Step  2: Find a most delayed vertex.
# [v]     Step 2.1: Find a max delayed flow (Thm C.6|Algorithm 1)
# [v]     Step 2.2: More specific max delayed flow (Prop 3.14)
# [≈]         Step 2.2.2: Iteratively bouw de g_k in de proof tot g_{|V|}.
# [v]     Step 2.3: If any maximum XY goto 3; otherwise goto 4.
# [ ] Step  3:
# [v]     Step 3.0: Vind maximal non-output XY v en w ∈ g[v] ⊆ O.
# [ ]     Step 3.1: CNOT(w, g[v] \ {w}) en daarna CNOT(N(v) ∩ O, w)
# [v]     Step 3.2: Buur van w is single vertex v. Extract w als hadamard en XY-msmnt angle, v nu nieuw frontier ipv w.
# [v] Step  4: Convert YZ to XY.
# [v]     Step 4.1: Pick maximal non-output v (is YZ) verbonden met output w.
# [~]     Step 4.2: Pivot+MBQC om vw. (Lem 2.32 + 4.5)
# [v]     Step 4.3: goto 1.
# [ ] Step  5: Nearly done: geen inner bipartite graaf meer.
# ( )     Step 5.1: Alle frontiers zijn nu triviaal extractbaar.
# ( )     Step 5.2: Input LCs zijn nu triviaal extractbaar.
# [ ]     Step 5.3: Fix evt permutaties met SWAPs.

def simplify_circuit(circ: Circuit) -> Circuit:
    """Does the full "There and Back Again" processing on a given input circuit.
    """
    # Step -1 --- I really can't be bothered to do this one.
    graph = cast(Graph, circ.to_graph())
    simplify.clifford_simp(graph)

    # Step 0
    graph = phase_gadgetify(graph)
    # Step 1
    in_progress = handle_output_lc(graph)
    # Loop step 2~4
    while True:
        (in_progress, done) = progress_extraction(in_progress)
        if done:
            break
    # Step 5
    return handle_input_lc(in_progress)

# Step 0
def phase_gadgetify(graph: Graph) -> Graph:
    """ MBQC+LC → Phase Gadget conversion

    Converts a circuit (promised to be in MBQC+LC form) into phase gadget form
    as described by Proposition 4.16.
    """
    og = OpenGraph(graph)
    # Complexity "O(no!)" but eh.
    updated = True
    while updated:
        updated = False
        for e in og.graph.edges():
            u = e[0]
            v = e[1]
            # Require both end points to be YZ, otherwise there's no
            # reason to separate them.
            if u in og.outputs or v in og.outputs or not (og.planes[u] == og.planes[v] == "YZ"):
                continue
            graph = pivot_mbqc(graph, e)
            updated = True
            break
        if updated:
            # I don't actually have to rebuild the og but I'm lazy.
            # The modifications are actually local.
            og = OpenGraph(graph)
            continue
        for u in og.graph.vertices():
            # Require an end point to be XZ, otherwise there's no
            # reason to convert it.
            if u in og.outputs or og.planes[u] != "XZ":
                continue
            graph = lcomp_mbqc(graph, u)
            # Same comment as above
            updated = True
        if updated:
            og = OpenGraph(graph)
            continue
    return graph

# Step 1
def handle_output_lc(graph: Graph) -> PartiallyExtracted:
    """ Phase Gadget → Beginning of extraction

    Ensures the frontier consists of unconnected phaseless Z spiders that are
    not mutually connected.
    In other words, this turns the LCs extracted and removes H-edges between
    the outputs.
    Phaselessness is guaranteed by the fact that the output verts are not
    measured.
    """
    pe = PartiallyExtracted(graph)
    pe.cleanup_outputs()
    return pe

# Step 2~4's loop
def progress_extraction(in_progress : PartiallyExtracted) -> Tuple[PartiallyExtracted, bool]:
    """ Does one iteration of the loop 2~4, and returns whether it's the final iter. """
    og = in_progress.open_graph
    max_verts = find_maximal_vertices(og)

    any_XY = False
    candidate_vertex = max_verts[0]
    for v in max_verts:
        plane = og.planes[v]
        if plane == "XY":
            any_XY = True
            candidate_vertex = v
            break
        elif plane == "XZ":
            raise ValueError(f"The graph at this point should not have XZ-measured vertices, but {v} is.")
        # Guaranteed YZ plane.
        # If this vertex is next to an output, make it a candidate.
        # (It is still overwritable if there are any XYs.)
        if set(og.outputs).intersection(og.graph.neighbors(v)):
            candidate_vertex = v

    if any_XY:
        in_progress = handle_maximal_XY(candidate_vertex, in_progress)
    else:
        in_progress = handle_maximal_YZ(candidate_vertex, in_progress)
    
    unextracted = in_progress.open_graph
    # TODO: Is this a valid way to check whether no internal verts?
    # How are identity wires handled?
    done = unextracted.graph.num_vertices() == unextracted.graph.num_inputs() + unextracted.graph.num_outputs()
    return (in_progress, done)

# Step 2
def find_maximal_vertices(og: OpenGraph) -> List[int]:
    """Returns all vertices that are maximal according to a most delayed gflow.
    """
    output = find_max_delayed_flow(og)
    if not output:
        raise ValueError("Malformed graph.")
    (g, d) = output
    (g, d) = focus_gflow(og, g, d)
    return [v for v in d.keys() if d[v] == 1]

# Step 3
def handle_maximal_XY(v: int, current_state: PartiallyExtracted) -> PartiallyExtracted:
    """Given a maximal XY vertex, processes the graph appropriately.

    Given a most-delayed-gflow-maximal XY-plane measured `vertex`, returns a
    partially extracted circuit where `vertex` is moved onto the frontier.

    This may introduce CNOTs during processing and will introduce one Hadamard.
    """
    og = current_state.open_graph
    # Grab neighbouring w ∈ g[v]
    gflow = find_max_delayed_flow(og)
    if not gflow:
        raise ValueError("We don't have gflow, wut?")
    (g, d) = gflow
    (g, d) = focus_gflow(og, g, d)
    gv = g[v]
    if len(gv.difference(og.outputs)) > 0:
        raise ValueError("Maximal non-output XY gflow vert ought to have only output verts in gflow...")
    gv = list(gv)
    w = gv[0]
    gv.remove(w)
    # First CNOT(w, g[v] \ {w}) to disconnect w from everything else.
    for u in gv:
        current_state.cnot_outputs(w, u)
    
    og = current_state.open_graph
    # Now, CNOT(N(v) ∩ O, w) to disconnect v from other outputs.
    nvo = set(og.graph.neighbors(v)).intersection(og.outputs).difference({w})
    for u in nvo:
        current_state.cnot_outputs(u, w)
    return current_state

# Step 4
def handle_maximal_YZ(vertex: int, current_state: PartiallyExtracted) -> PartiallyExtracted:
    """Given a maximal YZ vertex, process the graph appropriately.

    Given a most-delayed-gflow-maximal YZ-plane measured `vertex` that is
    connected with some output, and where
        `handle_maximal_XY`
    cannot be applied due to having no candidates, converts it into a XY-plane
    measured vertex (without introducing any new YZ measurements).
    """
    og = current_state.open_graph
    g = og.original_graph
    for w in og.graph.neighbors(vertex):
        if w in og.outputs:
            g = pivot_mbqc(g, (vertex, w))
            current_state.open_graph = OpenGraph(g)
            return current_state
    raise ValueError(f"Vertex {vertex} does not satisfy YZ pre-conditions!")

# Step 5
def handle_input_lc(graph: PartiallyExtracted) -> Circuit:
    """ Graph with no internal vertices → Full circuit

    After applying the above methods you will end up with a graph with only
    input and output vertices.
    However, the outputs may be permuted wrt the inputs, and there may still be
    LC inputs. This handles those final edge cases.
    """
    raise NotImplementedError()