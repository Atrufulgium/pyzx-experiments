# TODO: These heuristics assume some relation between spider count and output gates.
# That's not the case.
# Instead, do something like section 5.4 of the book; instead of starting with a graphlike
# slowly build a graphlike; in particular don't fuse!
# Also maybe look into simulated annealing and papers (20) -- (22).
# EDIT: apparantly "low H-edge count" is actually a good idea (see papers).
#       I just fucked up.

import heapq
import pyzx as zx
import pyzx.simplify as simplify
from copy import deepcopy
from fractions import Fraction
from pyzx.circuit import Circuit
from pyzx.graph.graph_s import GraphS as Graph
from pyzx.utils import VertexType, EdgeType, FractionLike, FloatInt
from typing import Callable, Dict, Iterable, List, Literal, Optional, Set, Tuple, cast
from graph_helpers import *

def interior_clifford_simp_alt(
    g : Graph,
    quiet : bool = False,
    lcomp_map = simplify.lcomp_simp,
    pivot_map = simplify.pivot_simp
    ) -> int:
    """ Does the same simplifications as `interior_clifford_simp` in a different order. """
    # Somewhat copied into the debug method
    #     `highlight_first_iter_possible_targets`
    simplify.spider_simp(g, quiet=quiet)
    simplify.to_gh(g, quiet=quiet)
    i = 0
    while True:
        j = simplify.id_simp(g, quiet=quiet) # identity removal
        j += simplify.spider_simp(g, quiet=quiet) # fusion
        # j += pivot_map(g, quiet=quiet)
        # j += lcomp_map(g, quiet=quiet)

        while True:
            k = pivot_map(g, quiet=quiet)
            j += k
            if k == 0: break

        j = simplify.id_simp(g, quiet=quiet) # identity removal
        j += simplify.spider_simp(g, quiet=quiet) # fusion

        while True:
            k = lcomp_map(g, quiet=quiet)
            j += k
            if k == 0: break
        if j == 0: break
        
        i += 1
    return i

def lcomp_simp_deepest_first(g : Graph, quiet : bool) -> int:
    """ This local complementation matches only a vertex closest to the output. """
    (_, lcomp_phase_done, _) = dijkstra(g, g.outputs())
    for u in lcomp_phase_done:
        iters = lcomp_single_vertex(g, u, 'lcomp_simp_deepest_first', quiet)
        if iters > 0:
            return iters
    return 0

def lcomp_simp_shallowest_first(g : Graph, quiet : bool) -> int:
    """ This local complementation matches only a vertex furthest from the output. """
    (_, lcomp_phase_done, _) = dijkstra(g, g.outputs())
    for u in reversed(lcomp_phase_done):
        iters = lcomp_single_vertex(g, u, 'lcomp_simp_shallowest_first', quiet)
        if iters > 0:
            return iters
    return 0

def lcomp_simp_few_edges(g : Graph, quiet : bool) -> int:
    """ This lcomp only matches the vertex that results in fewest edges. """
    heap = get_lcomp_effect_on_edges_heap(g)
    
    while len(heap) > 0:
        (_, u) = heapq.heappop(heap)
        iters = lcomp_single_vertex(g, u, 'lcomp_simp_few_edges', quiet)
        if iters > 0:
            return iters
    return 0

def lcomp_simp_many_edges(g : Graph, quiet : bool) -> int:
    """ This lcomp only matches the vertex that results the most edges. """
    heap = get_lcomp_effect_on_edges_heap(g, delta_map = lambda v : -v)
    
    while len(heap) > 0:
        (_, u) = heapq.heappop(heap)
        iters = lcomp_single_vertex(g, u, 'lcomp_simp_many_edges', quiet)
        if iters > 0:
            return iters
    return 0

def lcomp_simp_maintain_edges(g : Graph, quiet : bool) -> int:
    """ This lcomp only matches the vertex that results in most even distr edges/non-edges. """
    heap = get_lcomp_effect_on_edges_heap(g, delta_map = lambda v : abs(v))
    
    while len(heap) > 0:
        (_, u) = heapq.heappop(heap)
        iters = lcomp_single_vertex(g, u, 'lcomp_simp_maintain_edges', quiet)
        if iters > 0:
            return iters
    return 0

def lcomp_simp_introducing_most(g : Graph, quiet : bool) -> int:
    """ This lcomp only matches the vertex that introduces most lcomps/pivot candidates. """
    heap = get_lcomp_effect_on_candidates_heap(g)

    while len(heap) > 0:
        (_, u) = heapq.heappop(heap)
        iters = lcomp_single_vertex(g, u, 'lcomp_simp_introducing_most', quiet)
        if iters > 0:
            return iters
    return 0

def pivot_simp_deepest_first(g : Graph, quiet : bool) -> int:
    """ This pivot matches only the edge closest to the output. """
    (_, _, pivot_phase_done) = dijkstra(g, g.outputs())
    for e in pivot_phase_done:
        iters = pivot_single_edge(g, e, 'pivot_simp_deepest_first', quiet)
        if iters > 0:
            return iters
    return 0

def pivot_simp_shallowest_first(g : Graph, quiet : bool) -> int:
    """ This pivot matches only the edge furthest from the output.
    
    This actually has different semantics from `deepest_first` with a flipped
    graph. `deepest_first` cares about *any* vertex being deep. This cares
    about *both* vertices being shallow.
    
    This is actually an interesting difference, so I'm keeping it. """
    (_, _, pivot_phase_done) = dijkstra(g, g.outputs())
    for e in reversed(pivot_phase_done):
        iters = pivot_single_edge(g, e, 'pivot_simp_shallowest_first', quiet)
        if iters > 0:
            return iters
    return 0

def pivot_simp_few_edges(g : Graph, quiet : bool) -> int:
    """ This pivot only matches the edge that results in fewest edges. """
    heap = get_pivot_effect_on_edges_heap(g)
    
    while len(heap) > 0:
        (_, e) = heapq.heappop(heap)
        iters = pivot_single_edge(g, e, 'pivot_simp_few_edges', quiet)
        if iters > 0:
            return iters
    return 0

def pivot_simp_many_edges(g : Graph, quiet : bool) -> int:
    """ This pivot only matches the edge that results in the most edges. """
    heap = get_pivot_effect_on_edges_heap(g, delta_map = lambda v : -v)
    
    while len(heap) > 0:
        (_, e) = heapq.heappop(heap)
        iters = pivot_single_edge(g, e, 'pivot_simp_many_edges', quiet)
        if iters > 0:
            return iters
    return 0

def pivot_simp_maintain_edges(g : Graph, quiet : bool) -> int:
    """ This pivot only matches the edge that results in the most even distr edges/non-edges. """
    heap = get_pivot_effect_on_edges_heap(g, delta_map = lambda v : abs(v))
    
    while len(heap) > 0:
        (_, e) = heapq.heappop(heap)
        iters = pivot_single_edge(g, e, 'pivot_simp_maintain_edges', quiet)
        if iters > 0:
            return iters
    return 0

def pivot_simp_introducing_most(g : Graph, quiet : bool) -> int:
    """ This pivot only matches the edge that introduces the most lcomp/pivot candidates. """
    heap : List[Tuple[int, Tuple[int, int]]] = []
    
    while len(heap) > 0:
        (_, e) = heapq.heappop(heap)
        iters = pivot_single_edge(g, e, 'pivot_simp_introducing_most', quiet)
        if iters > 0:
            return iters
    return 0

def any_simp_few_edges(g : Graph, quiet : bool) -> int:
    """ Interleaves `lcomp_simp_few_edges` and `pivot_simp_few_edges`. """
    lcomp_heap = get_lcomp_effect_on_edges_heap(g)
    pivot_heap = get_pivot_effect_on_edges_heap(g)

    return try_run_best_lcomp_or_pivot(
        g,
        lcomp_heap, pivot_heap,
        'any_simp_few_edges (lcomp)', 'any_simp_few_edges (pivot)',
        quiet
    )

def any_simp_many_edges(g : Graph, quiet : bool) -> int:
    """ Interleaves `lcomp_simp_many_edges` and `pivot_simp_many_edges`. """
    lcomp_heap = get_lcomp_effect_on_edges_heap(g, delta_map = lambda v : -v)
    pivot_heap = get_pivot_effect_on_edges_heap(g, delta_map = lambda v : -v)
    
    return try_run_best_lcomp_or_pivot(
        g,
        lcomp_heap, pivot_heap,
        'any_simp_many_edges (lcomp)', 'any_simp_many_edges (pivot)',
        quiet
    )

def any_simp_maintain_edges(g : Graph, quiet : bool) -> int:
    """ Interleaves `lcomp_simp_maintain_edges` and `pivot_simp_maintain_edges`. """
    lcomp_heap = get_lcomp_effect_on_edges_heap(g, delta_map = lambda v : abs(v))
    pivot_heap = get_pivot_effect_on_edges_heap(g, delta_map = lambda v : abs(v))
    
    return try_run_best_lcomp_or_pivot(
        g,
        lcomp_heap, pivot_heap,
        'any_simp_maintain_edges (lcomp)', 'any_simp_maintain_edges (pivot)',
        quiet
    )

def any_simp_introducing_most(g : Graph, quiet : bool) -> int:
    """ Interleaves `lcomp_simp_introducing_most` and `pivot_simp_introducing_most`. """
    lcomp_heap = get_lcomp_effect_on_candidates_heap(g)
    pivot_heap = get_pivot_effect_on_candidates_heap(g)

    return try_run_best_lcomp_or_pivot(
        g,
        lcomp_heap, pivot_heap,
        'any_simp_introducing_most (lcomp)', 'any_simp_introducing_most (pivot)',
        quiet
    )

Distances = Dict[int, FloatInt]

def dijkstra(g : Graph, initial : Iterable[int]) -> Tuple[Distances, List[int], List[Tuple[int,int]]]:
    """ Calculates distances to `initial` in `g`.
    
    BOUNDARY vertices are ignored and get a distance of `Infinity`. Unreachable
    vertices too. The other vertices get the minimum number of edges between
    them and any vertex int he `initial` set.

    The initial set may contain BOUNDARY vertices however.

    Apart from the distances list, also returns as 2nd element the vertices
    of ±π/2 phase not next to a boundary, ordered by depth. (At the same depth,
    it's arbitrary.) Similarly, as the 3rd element of the tuple it returns
    edges with both ends phase 0 or π ordered by depth (smallest of the two).
    """
    depths : Distances = dict()
    # To allow updates in the queue (e.g. turning (2,v) into (1,v)), store the
    # vertices we've done in this `done` set, which can then be ignored if seen
    # later.
    # The todo list is a `heapq`, which is not defined as a class but as a
    # bunch of methods.
    # I'm not happy with implementing dijkstra for the [n]th time.
    todo = []
    done = set()

    # These may have false positives, but do contain *everything* that its
    # corresponding rewrite rule can apply to.
    lcomp_phase_done : List[int] = []
    pivot_phase_done : List[Tuple[int,int]] = []
    pivot_phase_done_set : Set[Tuple[int,int]] = set()

    for v in g.vertices():
        depth : FloatInt
        vtype = g.type(v)
        if v in initial:
            depth = 0
        else:
            depth = float('inf')
        depths[v] = depth
        # Boundary verts may only be added if they're initial.
        if vtype != VertexType.BOUNDARY or v in initial:
            heapq.heappush(todo, (depth, v))

    while len(todo) > 0:
        (_, u) = heapq.heappop(todo)
        if u in done:
            continue
        done.add(u)

        u_depth = depths[u]
        # lcomps next to the boundary kinda don't work
        if has_lcomp_phase(g, u) and not is_adjacent_to_boundary(g, u):
            lcomp_phase_done.append(u)
        
        if has_pivot_phase(g, u):
            for v in g.neighbors(u):
                if not has_pivot_phase(g, v):
                    continue
                # ffs
                e1 = (u,v)
                e2 = (v,u)
                if e1 in pivot_phase_done_set or e2 in pivot_phase_done_set:
                    continue
                pivot_phase_done.append(e1)
                pivot_phase_done_set.add(e1)

        alt = u_depth + 1

        for v in g.neighbors(u):
            if g.type(v) == VertexType.BOUNDARY:
                continue
            if alt < depths[v]:
                depths[v] = alt
                heapq.heappush(todo, (alt, v))

    return (depths, lcomp_phase_done, pivot_phase_done)

def get_lcomp_effect_on_edges_heap(g : Graph, delta_map : Callable[[int],int] = lambda v : v) -> List[Tuple[int, int]]:
    """ Computes a heap prioritised by the differences an lcomp would make on edgecount.
    
    Given a graph `g`, goes through all spiders with phase ±π/2 and computes
    the effect of a local complementation. This is then put into the `heapq`
    heap this method returns.

    A `delta_map` can be provided to change the priority. """
    # A heap with (lcomp edge delta, vert id) pairs
    heap : List[Tuple[int, int]] = []

    for v in g.vertices():
        if not has_lcomp_phase(g, v):
            continue
        (neighbourhood_connectivity, full_count) = compute_neighbourhood_connectivity(g, v)
        neighbourhood_connectivity_after_lcomp = full_count - neighbourhood_connectivity
        delta = neighbourhood_connectivity_after_lcomp - neighbourhood_connectivity
        heapq.heappush(heap, (delta_map(delta), v))
    
    return heap

def get_pivot_effect_on_edges_heap(g : Graph, delta_map : Callable[[int],int] = lambda v : v) -> List[Tuple[int,Tuple[int,int]]]:
    """ Computes a heap prioritised by the diference a pivot would make on edgecount.
    
    Given a graph `g`, goes through all edges with both ends phase 0, π and
    computes the effect of a pivot. This is then put into the `heapq` heap this
    method returns.
    
    A `delta_map` can be provided to change the priority. """
    # A heap with (pivot edge delta, edge id) pairs
    heap : List[Tuple[int, Tuple[int,int]]] = []

    for (u,v) in g.edges():
        if not has_pivot_phase(g, u) or not has_pivot_phase(g, v):
            continue
        all_u_neighs = set(g.neighbors(u))
        all_v_neighs = set(g.neighbors(v))
        U = all_u_neighs.difference(all_v_neighs).difference({v})
        V = all_v_neighs.difference(all_u_neighs).difference({u})
        C = all_u_neighs.intersection(all_v_neighs)
        (present1, max1) = compute_bipartite_connectivity(g, U, V)
        (present2, max2) = compute_bipartite_connectivity(g, U, C)
        (present3, max3) = compute_bipartite_connectivity(g, V, C)
        present = present1 + present2 + present3
        max = max1 + max2 + max3
        present_after_pivot = max - present
        delta = present_after_pivot - present
        heapq.heappush(heap, (delta_map(delta), (u,v)))
    
    return heap

def get_lcomp_effect_on_candidates_heap(g : Graph) -> List[Tuple[int,int]]:
    """Computes a heap prioritised by the differences an lcomp would make on candidate count.
    
    Given a graph `g`, goes through all spiders with phase ±π/2 and computes
    the effect of a local complementation. This is then put into the `heapq`
    heap this method returns."""
    heap : List[Tuple[int, int]] = []

    for v in g.vertices():
        if g.type(v) == VertexType.BOUNDARY:
            continue
        if is_adjacent_to_boundary(g,v) and has_lcomp_phase(g, v):
            continue

        delta = 0

        for n in g.neighbors(v):
            if g.type(n) == VertexType.BOUNDARY:
                continue
            if is_adjacent_to_boundary(g,n) and has_lcomp_phase(g,n):
                continue

            if has_lcomp_phase(g, n):
                # Turning n's ±π/2 into 0,π removes 1 lcomp, and may introduce pivots
                delta -= 1
                for n2 in g.neighbors(n):
                    if has_pivot_phase(g, n2):
                        delta += 1
            elif has_pivot_phase(g, n):
                # Turning n's 0,π into ±π/2 adds 1 lcomp, and may remove pivots
                delta += 1
                for n2 in g.neighbors(n):
                    if has_pivot_phase(g, n2):
                        delta -= 1
            
            if not has_pivot_phase(g, n):
                continue
            # Complementing edges may also add/remove pivots
            for n2 in g.neighbors(v):
                if n <= n2:
                    continue
                if not has_pivot_phase(g, n2):
                    continue
                if g.connected(n, n2):
                    delta -= 1
                else:
                    delta += 1
        
        heapq.heappush(heap, (-delta, v))
    
    return heap

def get_pivot_effect_on_candidates_heap(g : Graph) -> List[Tuple[int, Tuple[int,int]]]:
    """ Computes a heap prioritised by the diference a pivot would make on candidate count.
    
    Given a graph `g`, goes through all edges with both ends phase 0, π and
    computes the effect of a pivot. This is then put into the `heapq` heap this
    method returns. """
    heap : List[Tuple[int, Tuple[int, int]]] = []

    for (u,v) in g.edges():
        if not has_pivot_phase(g, u) or not has_pivot_phase(g, v):
            continue

        delta = 0

        all_u_neighs = set(g.neighbors(u))
        all_v_neighs = set(g.neighbors(v))
        U = all_u_neighs.difference(all_v_neighs).difference({v})
        V = all_v_neighs.difference(all_u_neighs).difference({u})
        C = all_u_neighs.intersection(all_v_neighs)
        handled : Set[Tuple[int, int]] = set()
        pairs = [(U,V), (U,C), (V,C)]
        for (A,B) in pairs:
            for a in A:
                if not has_pivot_phase(g, a):
                    continue
                for b in B:
                    if not has_pivot_phase(g, b):
                        continue
                    if (a,b) in handled:
                        continue
                    if g.connected(a,b):
                        delta -= 1
                    else:
                        delta += 1
                    handled.add((a,b))
                    handled.add((b,a))
        
        heapq.heappush(heap, (-delta, (u,v)))
    
    return heap

def try_run_best_lcomp_or_pivot(
    g : Graph,
    lcomp_heap : List[Tuple[int,int]],
    pivot_heap : List[Tuple[int,Tuple[int,int]]],
    lcomp_name : str,
    pivot_name : str,
    quiet : bool
    ) -> int:
    """ Runs the best match of either of the two heaps. """

    while True:
        best_lcomp_candidate = (float('inf'), -1)
        best_pivot_candidate = (float('inf'), (-1,-1))
        if len(lcomp_heap) > 0:
            best_lcomp_candidate = lcomp_heap[0]
        if len(pivot_heap) > 0:
            best_pivot_candidate = pivot_heap[0]
        
        if best_lcomp_candidate[0] == float('inf') == best_pivot_candidate[0]:
            # No candidates left
            return 0
        
        # In case lcomp and pivot ties, pick lcomp. Generally smaller radius of
        # effect giving a larger chance the other is preserved.
        if best_lcomp_candidate[0] <= best_pivot_candidate[0]:
            (_,u) = heapq.heappop(lcomp_heap)
            iters = lcomp_single_vertex(g, u, lcomp_name, quiet)
            if iters > 0:
                print(f"Did lcomp {u} (score {_})")
                return iters
        else:
            (_,e) = heapq.heappop(pivot_heap)
            iters = pivot_single_edge(g, e, pivot_name, quiet)
            if iters > 0:
                print(f"Did pivot {e} (score {_})")
                return iters


def compute_neighbourhood_connectivity(g : Graph, v : int) -> Tuple[int, int]:
    """ Computes the number of connected vertices in `v`'s neighbourhood and the max.
    
    Given a graph `g`, goes through all of `v`'s neighbours and checks whether
    they are mutually connected.
    
    BOUNDARY vertices are disregarded in this calculation.
    
    The returned tuple contains (edges present, edges possible)."""
    neighbours = g.neighbors(v)
    return compute_bipartite_connectivity(g, neighbours, neighbours)

def compute_bipartite_connectivity(g : Graph, A : Iterable[int], B : Iterable[int]) -> Tuple[int, int]:
    """ Computes the number of edges from A to B, and the maximum possible.
    
    Given a graph `g` and two vertex sets `A` and `B`, checks whether each pair
    `(a,b)` is an edge in `g`. These sets may overlap.
    
    BOUNDARY vertices are disregarded in this calculation.
    
    The returned tuple contains (edges present, edges possible)."""
    connectivity : int = 0
    count : int = 0
    possible : int = 0
    # Don't double-count if A ∩ B ≠ ∅.
    counted : set[Tuple[int,int]] = set()
    for a in A:
        if g.type(a) == VertexType.BOUNDARY:
            continue
        count += 1

        for b in B:
            if g.type(b) == VertexType.BOUNDARY:
                continue
            if (a,b) in counted or a == b:
                continue
            connectivity += int(g.connected(a,b))
            possible += 1
            counted.add((a,b))
            counted.add((b,a))
    return (count, possible)

def lcomp_single_vertex(g : Graph, u : int, name : str, quiet : bool) -> int:
    """ Applies an lcomp on *at most* vertex `u`. 
    
    Returns the number of lcomps done. """
    return simplify.simp(
        g,
        name = name,
        match = simplify.match_lcomp_parallel,
        rewrite = simplify.lcomp,
        matchf = lambda v : v == u,
        quiet = quiet
    )

def pivot_single_edge(g : Graph, e : Tuple[int,int], name : str, quiet : bool) -> int:
    """ Applies a pivot on *at most* edge `e`.
    
    Returns the number of pivots done. """
    iters = simplify.simp(
        g,
        name = name,
        match = simplify.match_pivot_parallel,
        rewrite = simplify.pivot,
        matchf = lambda e2 : e == e2 or (e[1], e[0]) == e2,
        quiet = quiet 
    )
    if iters != 0:
        return iters
    return simplify.simp(
        g,
        name = name,
        match = simplify.match_pivot_boundary,
        rewrite = simplify.pivot,
        matchf = lambda e2 : e == e2 or (e[1], e[0]) == e2,
        quiet = quiet 
    )

def highlight_first_iter_possible_targets(
    c : Circuit,
    iters : set = {0},
    lcomp_map = simplify.lcomp_simp,
    pivot_map = simplify.pivot_simp
    ):
    """ Highlights all lcomp and pivot targets the first iteration sees.
    
    lcomp targets are highlighted as X spiders, pivot targets are highlighted
    as hadamard edges. The rest is Z spiders and regular edges. """
    g = cast(Graph, c.to_graph())
    g = deepcopy(g)
    
    simplify.spider_simp(g, quiet=True)
    simplify.to_gh(g, quiet=True)

    bound = max(iters)

    print("Possible lcomps and pivots at the iterations:\n  (lcomps are drawn as X spiders)\n  (vertices that lcomps turn into lcomps are drawn as Z spiders)\n  (pivots are drawn as H edges)")
    draw_graph : Graph | None = None

    for i in range(bound):
        j = simplify.id_simp(g, quiet=True) # identity removal
        j += simplify.spider_simp(g, quiet=True) # fusion
        
        if i in iters:
            if not draw_graph:
                draw_graph = __highlight(g)
            else:
                draw_graph @= __highlight(g)

        while True:
            k = pivot_map(g, quiet=True)
            j += k
            if k == 0: break

        j = simplify.id_simp(g, quiet=True) # identity removal
        j += simplify.spider_simp(g, quiet=True) # fusion

        while True:
            k = lcomp_map(g, quiet=True)
            j += k
            if k == 0: break
        if j == 0: break
    
    if not draw_graph:
        draw_graph = __highlight(g)
    else:
        draw_graph @= __highlight(g)
    zx.draw(draw_graph)

    print(g)

def __highlight(g : Graph) -> Graph:
    gclone = deepcopy(g)
    (_, lcomp_phase_done, pivot_phase_done) = dijkstra(g, g.outputs())
    lcomp_phase_done = set(lcomp_phase_done)
    pivot_phase_done = set(pivot_phase_done)

    to_remove = set()
    neighbours = set()

    for v in gclone.vertices():
        if gclone.type(v) == VertexType.BOUNDARY:
            to_remove.add(v)
        if v in lcomp_phase_done:
            gclone.set_type(v, VertexType.X)
            neighbours.update(gclone.neighbors(v))
        else:
            gclone.set_type(v, VertexType.BOUNDARY)
    
    for v in neighbours:
        if has_pivot_phase(g, v):
            gclone.set_type(v, VertexType.Z)

    gclone.remove_vertices(to_remove)
        
    for e in gclone.edges():
        if e in pivot_phase_done or (e[1], e[0]) in pivot_phase_done:
            gclone.set_edge_type(e, EdgeType.HADAMARD)
        else:
            gclone.set_edge_type(e, EdgeType.SIMPLE)

    return gclone