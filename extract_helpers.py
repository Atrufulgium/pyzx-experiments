from copy import deepcopy
from fractions import Fraction
from pyzx.graph.graph_s import GraphS as Graph
from pyzx.linalg import Mat2
import pyzx.rules as rules
import pyzx.simplify as simplify
from pyzx.utils import VertexType, EdgeType, FractionLike, FloatInt
from typing import Dict, List, Literal, Set, Tuple, cast
from graph_helpers import *

class Measurement:
    """ Represents a measurement on one of the Bloch planes in a Graph.
    
    A measurement on the Bloch plane is defined by one of the planes XY, XZ, or
    YZ. The phase is α where πα ∈ [0, 2π]; i.e. radians without the π factor.
    
    This also stores all vertices relevant to the measurement. This order should
    be from least deep to deepest in the graph. For instance, if you have an XZ
    measurement `=(u)-(v)-((w))`, with `v` the π/2 Z-spider and `w` the
    X-spider, the order should be `[u,v,w]`."""

    plane : Literal[None, "XY", "XZ", "YZ"]

    def __init__(self, plane : Literal[None, "XY", "XZ", "YZ"], phase : FractionLike = 0, vertices : List[int] = []):
        self.plane = plane
        self.phase = phase
        self.vertices = vertices
    
    def __repr__(self):
        if self.plane == None:
            return "Not a valid measurement"
        plane_string = self.plane
        phase_string = str(self.phase) if self.phase != 1 else ""
        verts_string = "→".join([str(v) for v in self.vertices])
        return f"{plane_string} measurement with phase {phase_string}π on verts {verts_string}"

def try_get_measurement(g: Graph, v: int, out: List[Measurement], throw_on_multiple : bool = True) -> bool:
    """ With `v` a vertex in graph `g`, tries to get `v`'s measurement.

    If `v` is a measured vertex in the graph, this returns `True` and puts the
    result into `out[0]`. Otherwise, it will return `False`. (Note that `out`
    is automatically cleared beforehand when called.)

    This signature enables code of the form:
    ```
    out : List[Measurement] = []
    if try_get_measurement(g, v, out):
        measurement = out[0]
        # do something with `measurement`
    ```

    The graph must contain only Z-spiders, X-spiders, regular edges, and
    Hadamard-edges.

    `v` is a measured vertex if:
    - `v` is of the form `=(α)=`, which is a non-zero XY(α) measurement;
    - `v` is of the form `=(v)-(α)`, which is an XY(α) measurement;
    - `v` is of the form `=(v)-((α))`, which is a YZ(α) measurement. Note that
    this is a phase gadget, and may also be written as `=(v)--(α)`;
    - `v` is of the form `=(v)-(π/2)-((α))`, which is an XZ(α) measurement.

    (Here `(v)` represents a Z-spider, `((v))` an X-spider, `=` an arbitrary
    number of edges (Hadamard or not), `-` a single non-Hadamard edge, and
    `--` a single Hadamard-edge.)

    It may be the case that `v` satisfies multiple of the above cases (or the
    same case multiple times). If so, this method raises an error. This
    *usually* indicates a malformed graph (or very unoptimised graph where you
    can still fuse a bunch of vertices). This behaviour can be disabled by
    setting `throw_on_multiple` to False. In this case, `out` contains all
    possible interpretations instead of just one.

    Note that `v` is not a measured vertex if any of the following hold:
    - `v` is `-(α)` or `-((α))`: of arity 1 via a regular edge;
    - `v` is `=(u)--(α)`: arity 1 Z connected to a Z via a H-edge;
    - `v` is `-(π/2)-((α))`: the middle node of a XZ-measurement.

    Without these conditions, the component parts of measurements would also
    be reported as measurements, which is undesirable.

    Note that in particular 0-phase X-spiders are not seen as measured, and
    output vertices are considered. Both of these are easily checked in
    consumer code to handle in the alternate way you want.
    """
    # None of the listed cases (as it's an X spider)
    if (g.type(v) != VertexType.Z):
        return False
    
    # None of the listed cases (component of larger measurement structure)
    ns = list(g.neighbors(v))
    v_arity = len(ns)
    if v_arity == 1 and g.edge_type((v, ns[0])) == EdgeType.SIMPLE:
        return False
    if v_arity == 1 and g.edge_type((v, ns[0])) == EdgeType.HADAMARD and g.type(ns[0]) == VertexType.Z:
        return False
    # (Obnoxiously nearly duplicated below)
    if (v_arity == 2 and g.type(v) == VertexType.Z and g.phase(v) == Fraction(1,2)
        and g.edge_type((v, ns[0])) == g.edge_type((v, ns[1])) == EdgeType.SIMPLE):
        for i in range(2):
            if g.type(ns[i]) == VertexType.X and len(g.neighbors(ns[i])) == 1:
                return False
    
    out.clear()
    # Can finally start the positive checks
    # Case 1
    if (g.phase(v) != 0):
        out.append(Measurement("XY", g.phase(v), [v]))

    # Neighbour-based cases 2~4
    for n in ns:
        nns = list(g.neighbors(n))
        n_arity = len(nns)
        # Cases 2, 3
        if n_arity == 1 and g.edge_type((v, n)) == EdgeType.SIMPLE:
            match g.type(n):
                case VertexType.Z: out.append(Measurement("XY", g.phase(n), [v, n]))
                case VertexType.X: out.append(Measurement("YZ", g.phase(n), [v, n]))
                case VertexType.BOUNDARY: pass
                case _: raise ValueError("The given graph contains more than just Z-spiders, X-spiders, regular edges, and Hadamard edges.")
        # Case 3 (Phase gadget variant)
        if n_arity == 1 and g.edge_type((v, n)) == EdgeType.HADAMARD and g.type(n) == VertexType.Z:
            out.append(Measurement("YZ", g.phase(n), [v, n]))
        # Case 4
        # (Obnoxiously duplicate from the negative check earlier, doesn't feel worth to extract)
        if n_arity == 2 and g.type(n) == VertexType.Z and g.phase(n) == Fraction(1,2):
            # Still need a bunch of checks but that depends on the neighbour's neighbour
            nn = nns[1] if nns[0] == v else nns[0]
            nn_arity = len(g.neighbors(nn))
            if nn_arity == 1 and g.type(nn) == VertexType.X and g.edge_type((v, n)) == g.edge_type((n, nn)) == EdgeType.SIMPLE:
                out.append(Measurement("XZ", g.phase(nn), [v, n, nn]))

    if throw_on_multiple and len(out) > 1:
        err = "\n  ".join([str(msmnt) for msmnt in out])
        raise ValueError(f"The given vertex {v} can be interpreted as measurement in multiple ways:\n  {err}\nHave you ensured you have fused what you can and the existence of gflow?")
    
    return len(out) >= 1

GFlow = Dict[int, Set[int]]
Depth = Dict[int, int]

class OpenGraph:
    """ Represents the (G,I,O,λ) tuple as in There and Back again.
    Also stores the measurement angles.
    
    The regular Graph class is a bit too direct: the measurements are actual
    vertices, and there's some useless input/outputs. This gets rid of
    everything and stores the extra data instead in separate lists.

    Can only be applied to MBQC diagrams.
    TODO: Extend to MBQC+LC.
    """
    def __init__(self, g : Graph):
        self.inputs : List[int] = []
        self.outputs : List[int] = []
        self.planes : Dict[int, Literal['XY','XZ','YZ']] = {}
        self.phases : Dict[int, FractionLike] = {}
        # Each input qubit maps to a list of arity-0 spiders that represent
        # what actions are applied to that qubit, in order from start to graph.
        # The arity-0 spiders are represented as VertexType and their phase.
        # Also stores the original vertex number.
        self.input_lcs : Dict[FloatInt, List[Tuple[int, FractionLike, int]]] = {}
        # Same as input_lcs, in order from graph to end.
        self.output_lcs : Dict[FloatInt, List[Tuple[int, FractionLike, int]]] = {}
        self.original_graph = g
        self.original_graph_measurement_artifacts : Dict[int, List[int]] = {}

        g = deepcopy(g)

        # Does python care whether you update what you're iterating?
        # I hope so.
        to_remove : List[int] = []

        # Remove all boundary vertices, and make their neighbours IO.
        # Note that indices don't get updated or anything, gaps just appear.
        for (g_puts, self_puts, self_put_lcs) in [(g.inputs(), self.inputs, self.input_lcs), (g.outputs(), self.outputs, self.output_lcs)]:
            for v in g_puts:
                qubit = g.qubit(v)
                self_put_lcs[qubit] = []
                # Anything with arity 2 counts as local and is not the actual edge
                # of the graph, so iterate until something with arity >2.
                # (Edge case: The input and output are connected by arity ≤2.
                #  Because of that, check whether "to_remove"'d already.)
                neighs = list(g.neighbors(v))
                if len(neighs) != 1:
                    raise ValueError(f"In/output {v} of the graph is not of arity 1.")
                # (Note: also move the new output to the qubit of the old output.)
                old_output_qubit = g.qubit(v)

                w = neighs[0]

                # Note: It may be tempting to do more than just the H that may
                # be there, but that gets hairy fast and does double work that
                # another place also does.

                if g.edge_type((v, w)) == EdgeType.HADAMARD:
                    self_put_lcs[qubit].append((VertexType.H_BOX, 0, v))
                to_remove.append(v)
                self_puts.append(w)

                # Do the move mentioned at the beginning to make sure outputs
                # remain on their proper row
                g.set_qubit(w, old_output_qubit)
                self.original_graph.set_qubit(w, old_output_qubit)

        # Note that we have added things in order of "extreme -> graph", but we
        # want in order of time, so we have to flip the output case.
        for v in g.outputs():
            qubit = g.qubit(v)
            self.output_lcs[qubit].reverse()
        
        # Remove all nodes that only facilitate measurement.
        # Also register all measurements in the meantime.
        for v in g.vertices():
            out : List[Measurement] = []
            if try_get_measurement(g, v, out):
                measurement = out[0]
                to_remove.extend(measurement.vertices[1:])
                if (measurement.plane): # guaranteed true but eh
                    self.planes[v] = measurement.plane
                    self.phases[v] = measurement.phase
                    self.original_graph_measurement_artifacts[v] = measurement.vertices
            elif v not in self.outputs and not g.type(v) == VertexType.BOUNDARY:
                # try_get_measurement does not consider phase-0 X spiders as
                # measured, while these graphs do.
                self.planes[v] = "XY"
                self.phases[v] = 0
                self.original_graph_measurement_artifacts[v] = [v]
        
        for v in self.planes.keys():
            if v in self.outputs:
                print("Problem graph:")
                zx.draw(g, labels=True)
                raise ValueError(f"Output vertex {v} is measured {self.planes[v]}({self.phases[v]})). This is not allowed.\n(Debug info:)\n  Inputs: {self.inputs}\n  Outputs: {self.outputs}")
        
        g.remove_vertices(to_remove)
        # Make images neater by not listing phases.
        for v in g.vertices():
            g.set_phase(v, 0)
        self.graph = g
    
    def __repr__(self):
        return f"Open graph {self.graph} with:\n  -Inputs  {self.inputs}\n  -Outputs {self.outputs}\n  -Planes {self.planes}\n  -Phases {self.phases}\n  -Input LCs {self.input_lcs}\n  -Output LCs {self.output_lcs}"


def find_max_delayed_flow(og : OpenGraph, continue_on_failure : bool = False, quiet : bool = True) -> Tuple[GFlow, Depth] | None:
    """ Get a maximum delayed extended gflow on a given graph, if it exists.

    The first returned argument is the `graph` function in the definition of
    gflow. The second returned argument can be used to recreate the `≺`; the
    closer you are to the output, the lower it gets, and each step it decreases
    strictly.

    The output is `None` if there is no gflow.

    This graph needs to be in proper form:
    - Outputs (or rather, the Z-spiders before the outputs) are not measured;
    - Internal vertices and inputs (Z-spiders after the inputs) are measured;
    - All internal edges are Hadamard;
    - The actual pyzx graph representation's inputs/outputs are arity 1;
    - The actual pyzx graph representation's inputs/outputs are not connected.

    This is procedure `GFLOW` of Algorithm 1 of There and Back Again.

    If `continue_on_failure` is true, vertices that cannot have correction sets
    will be assigned `{-1}` instead.
    """
    g : GFlow = {}
    d : Depth = {}
    outputs = set(og.outputs)
    
    # Outputs have depth 0.
    for o in outputs:                                       # for all v ∈ O do
        d[o] = 0                                            # d(v) ← 0
    return __gflow_aux(og, 1, outputs, g, d, continue_on_failure, quiet=quiet)


def __gflow_aux(og : OpenGraph, k : int, outputs : Set[int], g : GFlow, d : Depth, continue_on_failure : bool = False, quiet : bool = True) -> Tuple[GFlow, Depth] | None:
    """ Helper method for `find_max_delayed_flow`.

    The arguments share names with that method. The exceptions here are
    `outputs`, which deviates from the graph's listed outputs; we're working
    our way back and effectively shrinking the graph each iteration; and
    `inputs`, which doesn't change but is also different from the pyzx graph's
    inputs.

    This is procedure `GFLOWAUX` of Algorithm 1 of There and Back Again.
    """
    graph = og.graph
    # Go through all non-`output` vertices and see how many we can additionally
    # gflow to grow the progress. If the progress cannot grow further, we are
    # done, perhaps too soon.
    correction_candidates = outputs.difference(og.inputs)   # O' ← O \ I (recall gflow's g: not-O to not-I)
    non_outputs = [v for v in graph.vertices() if not v in outputs]
    correctable : Set[int] = set()                          # C ← ∅

    # Custom behaviour in case we allow unsolvables.
    # Really, check the commit delta to make it really make sense.
    unsolvable : Set[int] = set()

    # Try to extend the gflow to every not-yet-gflow'd vertex.
    for u in non_outputs:                              # for all u ∉ O do
        # Note that this throws if the plane doesn't exist, which we want.
        planes : Literal["XY", "XZ", "YZ"]
        try:
            plane = og.planes[u]
        except:
            raise ValueError(f"Graph contains unmeasurable non-output vertex {u}")
        correction : Set[int] = set()                       # K' ← ∅ (implicit)
        correction_candidates_list = list(correction_candidates)
        
        # Super memory inefficient to build the same thing a bazillion times, whatever
        A = Mat2([[1 if graph.connected(row, col) else 0 for col in correction_candidates_list] for row in non_outputs])

        if plane == "XY":                                   # if λ(u) = XY then
            b = Mat2([[1 if u == row else 0] for row in non_outputs])
            x = A.solve(b)
            if x:
                correction = {correction_candidates_list[i] for i in range(len(correction_candidates_list)) if x.data[i][0] == 1}
        elif plane == "XZ":                                 # else if λ(u) = XZ then
            b = Mat2([[1 if u == row or graph.connected(u, row) else 0] for row in non_outputs])
            x = A.solve(b)
            if x:
                correction = {correction_candidates_list[i] for i in range(len(correction_candidates_list)) if x.data[i][0] == 1}
                correction.add(u)
        else: # plane == "YZ"                               # else λ(u) = YZ
            b = Mat2([[1 if graph.connected(u, row) else 0] for row in non_outputs])
            x = A.solve(b)
            if x:
                correction = {correction_candidates_list[i] for i in range(len(correction_candidates_list)) if x.data[i][0] == 1}
                correction.add(u)

        if len(correction) > 0:                             # if K' exists then
            correctable.add(u)                              # C ← C ∪ {u}
            g[u] = correction                               # g(u) ← K'
            d[u] = k                                        # d(u) ← k
        # Custom behaviour on `continue_on_failure`
        # Non-correctable vertices are by definition maximal in the order.
        # So after this, these being put into the new "outputs" layer is fine.
        elif continue_on_failure:
            unsolvable.add(u)

    # Custom behaviour: if there's no solvables, but there are *unsolvables*,
    # we may still be able to continue by assigning a unsolvable as output of a
    # new layer.
    # (By definition, vertices without correction set are maximal in the order.)
    if continue_on_failure and len(correctable) == 0 and len(unsolvable) > 0:
        if not quiet:
            print(f"Unsolvable; choices for lacking vert: {unsolvable}")
        u = unsolvable.pop()
        correctable.add(u)
        g[u] = {-1}
        d[u] = k
    
    # We now know what can be corrected, if any, so time for post processing to
    # the next depth iteration, if possible.
    if len(correctable) == 0:                               # if C = ∅ then
        done_size = len(outputs)
        # OpenGraph removed the extra stuff so this is fine.
        graph_size = graph.num_vertices()
        if done_size == graph_size:                         # if O = V(G) then
            return (g, d)                                   # return (true, g, d)
        else:
            return None                                     # return (false, ∅, ∅)
    return __gflow_aux(og, k+1, outputs.union(correctable), g, d, continue_on_failure, quiet)

def odd_neighbourhood(og : OpenGraph, verts : Set[int]) -> Set[int]:
    """ Computes the odd neighbourhod of a set in a graph. 
    
    This method is only well-defined for open graphs as in There and Back Again
    as otherwise measurement artifacts would be taken into account.
    """
    result : Set[int] = set()
    for v in verts:
        result = result.symmetric_difference(og.graph.neighbors(v))
    return result

def augment_gflow(v : int, w : int, g : GFlow, d : Depth) -> Tuple[GFlow, Depth]:
    """ Given v ≺ w in a graph, augments the gflow at v into g[v] △ g[w].

    Note that gflow guarantees v ≺ w iff d[w] < d[v], but otherwise it is only
    necessary but not sufficient.

    The original data is not modified. This maintains gflow.

    This is lemma 3.12 of There and Back Again.
    """
    g2 = deepcopy(g)
    g2[v] = g[v].symmetric_difference(g[w])
    return (g2, d)

def structure_gflow(og : OpenGraph, v : int, g : GFlow, d : Depth) -> Tuple[GFlow, Depth]:
    """ Makes the gflow at vertex v a bit more structured.

    In particular, after this method, at most g[v] is updated such that:
    - Non-output non-`v` vertices in g[v] are XY-measured;
    - Non-output non-`v` vertices in Odd(g[v]) are not XY-measured.

    The original data is not modified. This maintains gflow.

    This is lemma 3.13 of There and Back Again.
    """
    gk = deepcopy(g)
    if v in og.outputs:
        raise ValueError("Tried to change planes of an output vert, which doesn't make sense.")
    
    illegal = {v}.union(og.outputs)
    while True:
        set_XY : Set[int] = {u for u in odd_neighbourhood(og, gk[v]) if not u in og.outputs and og.planes[u] == "XY"}
        set_XY = set_XY.difference(illegal)
        set_XZ : Set[int] = {u for u in gk[v] if not u in og.outputs and og.planes[u] == "XZ"}
        set_XZ = set_XZ.difference(illegal)
        set_YZ : Set[int] = {u for u in gk[v] if not u in og.outputs and og.planes[u] == "YZ"}
        set_YZ = set_YZ.difference(illegal)

        s = set_XY.union(set_XZ).union(set_YZ)
        if not s:
            return (gk, d)
        
        # "Choose [augment target] w ∈ s among elements minimal in ≺."
        max_depth = max([d[w] for w in s])
        w = [w for w in s if d[w] == max_depth][0]
        (gk, _) = augment_gflow(v, w, gk, d)

def focus_gflow(og : OpenGraph, g : GFlow, d : Depth) -> Tuple[GFlow, Depth]:
    """ Makes gflow pretty nice wrt XY measurements.

    Turns a given gflow into one such that for all `v`:
    - Non-output non-`v` vertices in g[v] are XY-measured;
    - Non-output non-`v` vertices in Odd(g[v]) are not XY-measured.

    The original data is not modified and the partial order remains the same.
    In particular, a maximally delayed flow stays maximally delayed.

    This is lemma 3.14 of There and Back Again.
    """
    for v in og.graph.vertices():
        if v in og.outputs:
            continue
        (g, d) = structure_gflow(og, v, g, d)
    return (g, d)

def verify_gflow(og : OpenGraph, g : GFlow, d : Depth) -> bool:
    """ Checks the given gflow is actually a gflow.

    With g the correction sets, and ≺ defined by the transitive closure of
    `a`≺`b` iff `d[a] > d[b]` for `a`~`b` in the graph, checks all
    requirements of gflow for (g,≺).

    (Of course, these g and d are the output of `find_max_delayed_flow()`.)
    """
    check_order : List[List[int]] = []
    i = 1
    while True:
        depth_i = [v for v in d.keys() if d[v] == i]
        i += 1
        if len(depth_i) > 0:
            check_order.insert(0, depth_i)
        else:
            break

    past : Set[int] = set()
    for list in check_order:
        # Don't allow vertices on the same layer to refer to eachother.
        # (Exception for each vertex: itself.)
        past = past.union(list)
        for v in list:
            # (g1): correction set lie in the non-strict future
            for w in g[v]:
                if w in past and w != v:
                    return False
            # (g2): odd neighbourhood lies in the non-strict future
            odd = odd_neighbourhood(og, g[v])
            for w in odd:
                if w in past and w != v:
                    return False
            
            plane = og.planes[v]
            if plane == "XY":
                # (g3): v not in g(v), v in Odd(g(v))
                if v in g[v] or v not in odd:
                    return False
            elif plane == "XZ":
                # (g4): v in g(v), v in Odd(g(v))
                if v not in g[v] or v not in odd:
                    return False
            else: # plane == "YZ"
                # (g5): v in g(v), v not in Odd(g(v))
                if v not in g[v] or v in odd:
                    return False
    return True

def replace_measurement(g : Graph, v : int, plane : Literal["XY", "XZ", "YZ"], phase : FractionLike, existing_artifact : List[int]):
    """ Changes a measurement in a graph in-place.
    
    Given a vertex `v` in `g`, replaces its measurement artifact (as reported
    OpenGraph) with a new artifact. They look as follows:
    - With `XY`, you get a `-(v)` back;
    - With `XZ`, you get a `-(v)-(π/2)-((α))` back;
    - With `YZ`, you get a `-(v)--(α)` back.

    You *have* to pass the list of the current existing artifacts, again as
    reported by OpenGraph.

    Of course, this changes labels in the graph, but `v`'s label is guaranteed
    to not change by this process (if input is not malformed).
    """
    if existing_artifact[0] != v:
        raise ValueError(f"The artifact {existing_artifact} does not belong to vertex {v}.")

    old_count = len(existing_artifact)
    new_count = {"XY": 1, "XZ": 3, "YZ": 2}[plane]
    # We can add the new one nicely if we go from 2 to 3.
    # We have to just wing it if we go from 1 to *.
    if new_count > old_count:
        if old_count == 2:
            newq = 2 * g.qubit(existing_artifact[1]) - g.qubit(existing_artifact[0])
            newr = 2 * g.row(existing_artifact[1]) - g.row(existing_artifact[0])
            new = add_vertex(g, newq, newr, VertexType.X, 0)
            g.add_edges([(new, existing_artifact[-1])], EdgeType.SIMPLE)
            existing_artifact.append(new)
        else:
            for i in range(new_count - old_count):
                newq = g.qubit(existing_artifact[0]) + 0.33 * (i+1)
                newr = g.row(existing_artifact[0]) + 0.33 * (i+1)
                new = add_vertex(g, newq, newr, VertexType.Z, 0)
                g.add_edges([(new, existing_artifact[-1])], EdgeType.SIMPLE)
                existing_artifact.append(new)
    elif new_count < old_count:
        for i in range(old_count - new_count):
            g.remove_vertex(existing_artifact[-1])
            del existing_artifact[-1]
        
    # Now "existing_artifact" contains the vertexes to work with.
    g.set_type(existing_artifact[0], VertexType.Z)
    if plane == "XY":
        g.set_phase(existing_artifact[0], phase)
    elif plane == "XZ":
        [u1,u2,u3] = existing_artifact
        g.set_type(u2, VertexType.Z)
        g.set_type(u3, VertexType.X)
        g.set_phase(u1, 0)
        g.set_phase(u2, Fraction(1,2))
        g.set_phase(u3, phase)
        g.set_edge_type((u1,u2), EdgeType.SIMPLE)
        g.set_edge_type((u2,u3), EdgeType.SIMPLE)
    else: # plane == "YZ"
        [u1,u2] = existing_artifact
        g.set_type(u2, VertexType.Z)
        g.set_phase(u1, 0)
        g.set_phase(u2, phase)
        g.set_edge_type((u1,u2), EdgeType.HADAMARD)

def pivot_mbqc(g : Graph, uv : Tuple[int,int]) -> Graph:
    """ Pivots a MBQC+LC graph around edge uv, and returns a MBQC+LC.

    If the input is not MBQC+LC, the result will also most likely not be.

    This is an application of first Lemma 2.32 and then Prop 4.5 of There and
    Back Again. This maintains gflow.

    The input graph remains untouched.

    NOTE: Currently unsupported when either u or v are XZ measured, or YZ-
    measured without Hadamard edge. This is not a problem in my use-cases
    however.
    """
    gclone = g.clone()
    iters = simplify.simp(
        gclone,
        name = "Single pivot",
        match = rules.match_pivot_parallel,
        rewrite = rules.pivot,
        matchf = lambda e : uv[0] in cast(Tuple[int,int], e) and uv[1] in cast(Tuple[int,int],e),
        quiet = False
    )
    if iters != 1:
        raise ValueError(f"Did not simplify one iteration, instead {iters}. Was this really pivot-able? Check whether both ends of uv have a π-multiple phase.\nYou may have also encountered a measured pivot vertex in a currently unsupported plane (XZ, or YZ via regular edge).")

    # NOTE: The lemma talks about the pivoteds being measured, but that seems
    # pretty unsupported in pyzx other than "XY aπ" or "YZ any via gadget".
    # However, we don't actually need all of that functinoality.
    # In any case, the above allowed cases may introduce a π factor in
    # neighbouring X spiders of both u and v.
    # (This also includes outputs, which needs some thought.)
    
    og = OpenGraph(g) # Need for easy access to plane data
    if (uv[0] in og.inputs or uv[1] in og.inputs):
        raise NotImplementedError(f"Think about this case. (Input in {uv})")
    for w in g.neighbors(uv[0]) | g.neighbors(uv[1]):
        if w in uv: # These don't exist anymore
            continue

        # When w is an output you need to be more considerate because the OG
        # doesn't recognise it as a measurement.
        plane : Literal['XY','YZ','XZ']
        msmnt : List[int]
        if w in og.planes:
            plane = og.planes[w]
            msmnt = og.original_graph_measurement_artifacts[w]
        else:
            plane = 'XY'
            msmnt = [w]

        if plane == "XY":
            # Cases =(α) and =(w)-(α).
            # Former is fine, latter needs w's phase added to α.
            if len(msmnt) == 2:
                gclone.set_phase(msmnt[1], gclone.phase(msmnt[1]) + gclone.phase(msmnt[0]))
                gclone.set_phase(msmnt[0], 0)
        elif plane == "XZ":
            # Case =(w)-(π/2)-((α)).
            # Flip α's phase.
            gclone.set_phase(msmnt[2], -gclone.phase(msmnt[2]))
            gclone.set_phase(msmnt[0], 0)
        elif plane == "YZ":
            # Cases =(w)-((α)) and =(w)--(α).
            # In either case, flip α's phase.
            gclone.set_phase(msmnt[1], -gclone.phase(msmnt[1]))
            gclone.set_phase(msmnt[0], 0)
        elif not gclone.type(w) == VertexType.BOUNDARY:
            raise ValueError("Malformed input, this is not a branch you should normally be able to reach.")

    return gclone

def lcomp_mbqc(g : Graph, u : int) -> Graph:
    """ Local complements a MBQC+LC graph around vert u, and returns a MBQC+LC.

    If the input is not MBQC+LC, the result will also most likely not be.

    This is an application of first Lemma 2.31 and then Prop 4.3 of There and
    Back Again. This maintains gflow.

    The input graph remains untouched. The output may have different
    measurement artifacts with different labels.

    This does *not* do the ±π/2 removal you usually expect from lcomps. The
    main purpose here is removal of annoying measurement planes.
    """
    gclone = g.clone()
    # This lcomp needs to be manual as the rewrite rules api doesn't seem to
    # want to work when measurements are attached. Simple enough.
    # (1) u gets a X π/2 spider so that measurements change:
    #     - XY(α) ↦ XZ(π/2 - α)
    #     - XZ(α) ↦ XY(α - π/2)
    #     - YZ(α) ↦ YZ(α + π/2)
    # (2) N(u) get a Z -π/2 spider so that measurements change:
    #     - XY(α) ↦ XY(α - π/2)
    #     - XZ(α) ↦ YZ(α)
    #     - YZ(α) ↦ XZ(-α)
    # (3) Flip all edges in N[u].

    og = OpenGraph(g)
    if (u in og.inputs):
        raise NotImplementedError(f"Think about this case. (Input {u})")
    if (u in og.outputs):
        raise ValueError(f"Ought to be unsupported, why did this happen? (Output {u})")
    # Note that while the labels update, the updates are independent.
    # So working on the same open graph the whole time is justified.
    # (1)
    u_phase = og.phases[u]
    u_artifact = og.original_graph_measurement_artifacts[u]
    match og.planes[u]:
        case "XY": replace_measurement(gclone, u, "XZ", Fraction(1,2) - u_phase, u_artifact)
        case "XZ": replace_measurement(gclone, u, "XY", u_phase - Fraction(1,2), u_artifact)
        case "YZ": replace_measurement(gclone, u, "YZ", u_phase + Fraction(1,2), u_artifact)
    # (2)
    for w in og.graph.neighbors(u):
        w_phase : FractionLike
        w_artifact : List[int]
        w_plane : Literal["XY", "XZ", "YZ"]
        # Outputs are also affected by the operation, even if they are not
        # usually considered measured.
        if w in og.outputs:
            w_phase = 0
            w_artifact = [w]
            w_plane = "XY"
        else:
            w_phase = og.phases[w]
            w_artifact = og.original_graph_measurement_artifacts[w]
            w_plane = og.planes[w]
        match w_plane:
            case "XY": replace_measurement(gclone, w, "XY", w_phase - Fraction(1,2), w_artifact)
            case "XZ": replace_measurement(gclone, w, "YZ", w_phase, w_artifact)
            case "YZ": replace_measurement(gclone, w, "XZ", -w_phase, w_artifact)
        # If we're an output, we introduced a phase so it no longer counts as
        # an output. Fix this.
        if w in og.outputs:
            # This is unique
            o = [v for v in g.neighbors(w) if v in g.outputs()][0]
            # Put a new opengraph-output halfway between.
            q_w = g.qubit(w)
            q_o = g.qubit(o)
            r_w = g.row(w)
            r_o = g.row(o)
            dq = q_o - q_w
            dr = r_o - r_w
            i = add_vertex(gclone, q_w + dq, r_w + dr, VertexType.Z, 0)
            j = add_vertex(gclone, q_w + 2 * dq, r_w + 2 * dr, VertexType.Z, 0)
            gclone.set_position(o, q_w + 3 * dq, r_w + 3 * dr)
            gclone.remove_edge((w,o))
            gclone.add_edges([(w,i),(i,j)], EdgeType.HADAMARD)
            gclone.add_edges([(j,o)], EdgeType.SIMPLE)
    # (3) -- hooray for the smart replace option
    for w in og.graph.neighbors(u):
        for v in og.graph.neighbors(u):
            if v == w:
                continue
            gclone.add_edges([(v,w)], EdgeType.HADAMARD, smart = True)
    
    return gclone

def normalize_graph(g : Graph) -> Graph:
    """ Makes all measurements of an MBQC+LC graph consistent.
    
    In particular, rewrites XY msmnts into 1 vert, and YZs into gadgets.
    
    Leaves the original graph untouched. """
    gclone = deepcopy(g)
    og = OpenGraph(g)
    for v in og.graph.vertices():
        if v in og.outputs:
            continue
        plane = og.planes[v]
        phase = og.phases[v]
        art = og.original_graph_measurement_artifacts[v]
        replace_measurement(gclone, v, plane, phase, art)
    return gclone

def highlight_missing_correction_set(graph : Graph):
    """ Draws all vertices without correction set as red X spiders instead.
    This will naturally include all output vertices and measurement artifacts.
    """
    graph = deepcopy(graph)
    flow = find_max_delayed_flow(OpenGraph(graph), True)
    if flow == None:
        raise ValueError("whoops")
    (g,d) = flow
    for v in graph.vertices():
        if graph.type(v) == VertexType.BOUNDARY:
            continue
        if not v in g or g[v].pop() == -1:
            graph.set_type(v, VertexType.X)
    zx.draw(graph, labels=True)