import random as rng
import math
import pyzx as zx
from pyzx.graph.graph_s import GraphS as Graph
from pyzx.utils import VertexType, EdgeType, FractionLike, FloatInt
from fractions import Fraction
from typing import Callable, List, Literal, Set, Tuple, cast

def single_vertex_graph(vertextype : int, phase : FractionLike = 0) -> Graph:
    """ Creates a single-vertex graph (not even input/outputs). """
    g = Graph()
    add_vertex(g, 0, 0, vertextype, phase)
    return g

def add_vertex(g : Graph, qubit : FloatInt, row : FloatInt, vertextype : int, phase : FractionLike | None, connections : List[int | Literal["I", "O"]] = []) -> int:
    """ Helper method for adding vertices to a graph. Returns its label.
    
    You can optionally specify existing vertices in the graph to connect with.
    If you provide `"I"`, it will generate a new input to connect this with, and
    if you provide `"O"`, it will generate a new output to connect this with.
    """
    [label] = list(g.add_vertices(1))
    g.set_position(label, qubit, row)
    g.set_type(label, vertextype)
    if phase != None:
        g.set_phase(label, phase)

    for c in connections:
        if c == "I":
            i = add_vertex(g, qubit, row - 1, VertexType.BOUNDARY, phase = None, connections = [label])
            inputs = [inp for inp in g.inputs()]
            inputs.append(i)
            g.set_inputs(inputs)
        elif c == "O":
            o = add_vertex(g, qubit, row + 1, VertexType.BOUNDARY, phase = None, connections = [label])
            outputs = [out for out in g.outputs()]
            outputs.append(o)
            g.set_outputs(outputs)
        else:
            g.add_edge((label, c), EdgeType.HADAMARD)
    return label

def insert_vertex(g : Graph, uv : Tuple[int,int], vertextype : int, phase : FractionLike | None, move_vertices = False, extra_hadamards = False) -> int:
    """ Helper method for inserting a vertex into the middle of an existing edge.
    
    Returns the label of the new vertex."""
    if not g.connected(uv[0], uv[1]):
        raise ValueError(f"Must insert a vertex into an existing edge; {uv} does not exist.")
    u = uv[0]
    v = uv[1]
    q : FloatInt
    r : FloatInt

    if move_vertices:
        qu = g.qubit(u)
        qv = g.qubit(v)
        ru = g.row(u)
        rv = g.row(v)
        dq = abs(qv - qu)
        dr = abs(rv - ru)
        q = qv
        r = rv
        for qubit in g.vertices():
            qq = g.qubit(qubit)
            rq = g.row(qubit)
            if (dq != 0 and (qq >= qu and qq >= qv))\
                or (dr != 0 and (rq >= ru and rq >= rv)):
                g.set_position(qubit, qq + dq, rq + dr)
    else:
        q = (g.qubit(u) + g.qubit(v)) * 0.5
        r = (g.row(u) + g.row(v)) * 0.5
    i = add_vertex(g, q, r, vertextype, phase)
    edge_type = g.edge_type(uv)
    # Semantic equality: normal ↦ normal,normal; and h-edge ↦ h-edge,normal.
    # With extra hads:   normal ↦ h-edge,h-edge; and h-edge ↦ normal,h-edge.
    g.remove_edge((u,v))
    if extra_hadamards:
        edge_type = EdgeType.HADAMARD if edge_type == EdgeType.SIMPLE else EdgeType.SIMPLE
        g.add_edges([(u,i)], edge_type)
        g.add_edges([(i,v)], EdgeType.HADAMARD)
    else:
        g.add_edges([(u,i)], EdgeType.SIMPLE)
        g.add_edges([(i,v)], edge_type)
    return i

def merge_vertices(g : Graph, e : Tuple[int,int]):
    """ Merges `e = (u,v)` into `u`. """
    if g.edge_type(e) != EdgeType.SIMPLE:
        print("Problem graph:")
        zx.draw(g, labels = True)
        raise ValueError(f"Can only merge simple, non-hadamard edges, but {e} is not simple!")
    (u,v) = e
    if g.type(u) != VertexType.Z or g.type(v) != VertexType.Z:
        raise ValueError(f"Can only merge Z-vertices with eachother, but {e} has non-Z vertices! (This also disallows BOUNDARY verts.)")
    
    to_remove : Set[Tuple[int,int]] = set()
    for w in g.neighbors(v):
        w_type = g.edge_type((v,w))
        to_remove.add((v,w))
        # don't add (u,u) lol
        if w == u:
            continue
        # `_smart` to also account for canceling subsequent CZs.
        g.add_edge_smart((u,w), edgetype = w_type)
    g.remove_edges(to_remove)
    
    g.set_phase(u, g.phase(u) + g.phase(v))
    g.remove_vertex(v)

def is_adjacent_to_input(g : Graph, v : int) -> bool:
    """ Whether a vertex is next to an input BOUNDARY vertex. """
    return get_any_adjacent_where(g,v, lambda g,v,n: n in g.inputs()) != None

def is_adjacent_to_boundary(g : Graph, v : int) -> bool:
    """ Whether a vertex is adjacent to a `VertexType.BOUNDARY` vertex. """
    return get_any_adjacent_where(g, v, lambda g,v,n: g.type(n) == VertexType.BOUNDARY) != None

def get_any_adjacent_where(g : Graph, v : int, filter : Callable[[Graph, int, int], bool]) -> int | None:
    """ Returns one arbitrary neighbour of `v` satisfying a lambda with input
    `g,v,neighbour`, or `None` if such a neighbour does not exist.
    """
    for neighbour in g.neighbors(v):
        if filter(g, v, neighbour):
            return neighbour
    return None

def get_any_adjacent_where_raise_if_none(g : Graph, v : int, filter : Callable[[Graph, int, int], bool]) -> int:
    """ Returns one arbitrary neighbour of `v` satisfying a lambda with input
    `g,b,neighbour`, and raises a ValueError if no such neighbour exists."""
    ret = get_any_adjacent_where(g, v, filter)
    if ret == None:
        raise ValueError(f"Expected {v} to have a neighbour {filter}, but this was not the case.")
    return ret

def get_all_adjacent_where(g : Graph, v : int, filter : Callable[[Graph, int, int], bool]) -> List[int]:
    """ Returns all neighbours of `v` satisfying a lambda with input `g,v,neighbour`."""
    res = []
    for neighbour in g.neighbors(v):
        if filter(g, v, neighbour):
            res.append(neighbour)
    return res

def get_adjacent_boundary(g : Graph, v : int, raise_on_multiple_boundaries : bool = True) -> int | None:
    """ If a vertex is adjacent to exactly one BOUNDARY vertex, return it. """
    boundaries = get_all_adjacent_where(g, v, lambda g,v,n: g.type(n) == VertexType.BOUNDARY)
    if len(boundaries) > 1:
        if raise_on_multiple_boundaries:
            raise ValueError(f"Vertex {v} is connected to multiple boundaries ({boundaries}). This is not allowed.")
        else:
            return None
    if len(boundaries) == 1:
        return boundaries[0]
    return None

def has_lcomp_phase(g : Graph, v : int) -> bool:
    """ Whether a vertex has phase ±π/2. """
    v_phase = g.phase(v)
    return v_phase == Fraction(1,2) or v_phase == Fraction(3,2)

def has_clifford_phase(g : Graph, v : int) -> bool:
    """ Whether a vertex has a π/2-multiple phase. """
    return has_pivot_phase(g, v) or has_lcomp_phase(g, v)

def has_pivot_phase(g : Graph, v : int) -> bool:
    """ Whether a vertex has phase 0 or π. """
    v_phase = g.phase(v)
    return v_phase == 0 or v_phase == 1

def get_flipped_edge_type(edge_type : int) -> int:
    """ Maps EdgeType.HADAMARD and EdgeType.SIMPLE to eachother. """
    if edge_type == EdgeType.SIMPLE:
        return EdgeType.HADAMARD
    elif edge_type == EdgeType.HADAMARD:
        return EdgeType.SIMPLE
    raise ValueError(f"Unknown edge type {edge_type}")

def flip_edge_type(g : Graph, e : Tuple[int,int]):
    """ Changes H-ness of an edge in the graph. """
    et = g.edge_type(e)
    et = get_flipped_edge_type(et)
    g.set_edge_type(e, et)

def n_choose_k_up_to_order(n : int, k : int):
    """Returns from the values `0, .., n-1` all ascending k-length arrays."""

    def increment_array(arr : List[int], n : int) -> bool:
        """Represents one iteration of the outer iterator."""
        len_arr = len(arr)
        from_last : int = 1
        while True:
            index = len_arr - from_last
            if arr[index] + 1 > n - from_last:
                # This index would go out of bounds
                from_last += 1
                if from_last > len_arr:
                    break
                continue
            # This index does not go out of bounds; set the +1-incrementally
            # increasing sequence from this index out.
            increase_wrt_index = arr[index] + 1 - index
            for i in range(index, len_arr):
                arr[i] = i + increase_wrt_index
            return True
        return False

    if k > n or k < 0:
        raise ValueError(f"n={n} choose k={k} with k outside [0,n] is not valid.")
    arr = list(range(k))
    yield arr
    if k == 0:
        return
    # Try to increment the last index.
    # If this would put it outside of [0,n-1], instead increment the second-to-
    # last index, and set the last index to that +1.
    # If this would put taht outside of [0,n-2], instead increment the
    # third-to-last, set the 2nd-to to that +1, and the last to that +2.
    # Etcetera.
    while True:
        if increment_array(arr, n):
            yield arr
        else:
            return

def completely_random_graph(
    input_count : int, internal_count : int, output_count : int | None = None,
    p_edge : float = 0, p_XZ : float = 0, p_YZ : float = 0,
    seed : int | None = None) -> Graph:
    """ Creates a random open graph without any further structure.
    
    The only guarantee is that the resulting graph is connected. After that,
    edges are filled in randomly according to `p_edge`.

    If `output_count` is unspecified, it is the same as `input_count`.
    
    Internal verts have measurement planes respectively XZ, YZ, XY with
    probabilities `p_XZ`, `p_YZ`, `1-p_XZ-p_YZ`.
    
    A seed can be given to seed the random number generator.
    """
    if p_XZ + p_YZ > 1 or p_XZ < 0 or p_YZ < 0:
        raise ValueError(f"Expected probabilities (p_XZ={p_XZ}, p_YZ={p_YZ}) to each lie in [0,1] and their sum to lie in [0,1]. This is not the case.")
    
    if seed != None:
        rng.seed(seed)
    
    if output_count == None:
        output_count = input_count
    
    output_count = cast(int, output_count)
    
    max_puts = max(input_count, output_count)
    # Put input BOUNDARY verts in row -1
    # Put inputs in row 0
    # Put internal vertices in a circle of diameter row 1~row max_puts
    # Put outputs in row 1 + max_puts
    # Put output BOUNDARY verts in row 2 + max_puts
    g = Graph()
    boundary_inps : List[int] = []
    verts : List[int] = [] # Make a spanning tree and randomly add edges between these
    boundary_outs : List[int] = []
    for i in range(0, input_count):
        u = add_vertex(g, i, -1, VertexType.BOUNDARY, phase = None)
        v = add_vertex(g, i, 0, VertexType.Z, 0)
        g.add_edge((u,v), EdgeType.SIMPLE)
        verts.append(v)
        boundary_inps.append(u)
    for i in range(0, internal_count):
        angle = i * 2 * math.pi / internal_count
        radius = (max_puts - 1) / 2
        if max_puts == 1:
            radius += 0.5
        (qubit, row) = (math.sin(angle) * radius * 0.9 + radius, math.cos(angle) * radius * 0.9 + 1 + radius)
        if max_puts == 1:
            qubit += 0.5
        phase = 0
        r = rng.random()
        v : int | None = None
        w : int | None = None
        # Random plane (measured π/4)
        if r < p_XZ:
            v = add_vertex(g, qubit + 0.1, row + 0.1, VertexType.Z, Fraction(1,2))
            w = add_vertex(g, qubit + 0.2, row + 0.2, VertexType.X, Fraction(1,4))
        elif p_XZ < r < p_XZ + p_YZ:
            v = add_vertex(g, qubit + 0.1, row + 0.1, VertexType.Z, Fraction(1,4))
        else:
            phase = Fraction(1,4)
        u = add_vertex(g, qubit, row, VertexType.Z, phase)
        if w != None:
            g.add_edges([(u,v), (v,w)], EdgeType.SIMPLE)
        elif v != None:
            g.add_edge((u,v), EdgeType.HADAMARD)
        verts.append(u)
    for i in range(0, output_count):
        u = add_vertex(g, i, 1 + max_puts, VertexType.Z, 0)
        v = add_vertex(g, i, 2 + max_puts, VertexType.BOUNDARY, phase = None)
        g.add_edge((u,v), EdgeType.SIMPLE)
        verts.append(u)
        boundary_outs.append(v)
    
    g.set_inputs(boundary_inps)
    g.set_outputs(boundary_outs)

    # First add a spanning tree to ensure connectivity.
    # https://mathoverflow.net/a/458370 2nd comment.
    vert_count = len(verts)
    edges : Set[Tuple[int,int]] = set() # BOTH pairs (u,v), (v,u)
    v_old = verts[rng.randint(0, vert_count-1)]
    visited : Set[int] = {v_old}
    while len(visited) < vert_count:
        v_new = v_old
        while v_new == v_old:
            v_new = verts[rng.randint(0, vert_count-1)]
        if not v_new in visited:
            edges.add((v_new, v_old))
            edges.add((v_old, v_new))
            visited.add(v_new)
        v_old = v_new
    
    # Now add random edges according to `p_edge`.
    for i in range(vert_count):
        for j in range(i + 1, vert_count):
            if rng.random() < p_edge:
                u = verts[i]
                v = verts[j]
                edges.add((u,v))
                edges.add((v,u))
    
    g.add_edges([e for e in edges if e[0] < e[1]], EdgeType.HADAMARD)
    return g