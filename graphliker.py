from typing import Callable
import pyzx.simplify as simplify
from pyzx.circuit import Circuit
from extract_helpers import *
from lcomping import lcomp_single_vertex
from graph_helpers import *

import pyzx as zx

# A global reset in `to_graphlike` and read/written from/to in child call `pivot_single_edge_boundary`
pivot_row : int = 0
# A global reset in `to_graphlike`, written to in `pivot_single_edge_boundary`, and again read from `to_graphlike`
gadgets : List[Tuple[int,int]] = []

def to_graphlike(circuit : Circuit, quiet : bool = True, dontdraw = True, verify_stepwise = False) -> Graph:
    """ Convert a circuit to a lower-vertex graphlike diagram. 
    
    This procedure differs from pyzx and instead does something more like
    [book]'s [tentative] Proposition 5.4.5.
    
    This means that instead of globally turning the graph into a graphlike by
    doing, in order, cc'ing red spiders, canceling Hs, fusing, removing self-
    loops, canceling H-edges, we instead move from the start of the circuit to
    the end incrementally building a graphlike diagram.

    The specific strategy here consists of:
    - First color change all red spiders and cancel Hs;
    - Insert --(0)--(0)- at each input.

    Then start moving a frontier from left to right.
    - If we encounter CZ/S without Hs in the way, simply absorb it;
    - If we encounter CZ/S with Hs in the way, lcomp/pivot;
    - If we encounter a non-Clifford, either absorb it, pivot, or leave it be.

    However, this is not sufficient. For instance, it does not process a line
    with inputs `non-Cliff HAD CZ HAD non-Cliff` into just one with merged
    phases, which you can do if you are willing to make CZs just vertical lines
    (which we aren't).

    So for now this gives up and runs the default ZX simplification algorithm.
    The upside is that the graph is *much* smaller (O(non-Cliffords) instead
    of O(gates)) so there is still some benefit in runtime.
    """
    g : Graph = cast(Graph, circuit.to_graph())
    simplify.to_gh(g) # Also does H-cancellation already
    frontier : List[int] = [] # Small, so list
    backrow : Set[int] = set() # No need to remove
    for i in g.inputs():
        ni = g.neighbors(i)
        if len(ni) != 1:
            raise ValueError(f"Input {i} is not connected to just 1 neighbour!")
        n = list(ni)[0]
        # Create i -- (left) -- (right) - (n)
        right = insert_vertex(g, (i,n), VertexType.Z, phase = 0, extra_hadamards = False)
        left = insert_vertex(g, (i, right), VertexType.Z, phase = 0, extra_hadamards = True)
        backrow.add(left)
        frontier.append(right)
        # Clean up the graph
        i_row = g.row(i)
        g.set_row(right, i_row)
        g.set_row(left, i_row - 1)
        g.set_row(i, i_row - 2)
    
    # Throughout this process, the frontier will containt all rightmost verts
    # of the GSLC. "Right" is well-defined as we're still basically a circuit.
    # The right neighbours of the frontier are what we will be adding.
    # Note that by pivoting, we eventually get internal vertices.
    # The considered gates will be in order of row, selecting one at a time.
    # Naming convention: i,j are *indices* into `frontier`.
    #                    ni,nj are *indices* into `frontier` neighbouring i,j in g.
    #                    Starting with f are actual spiders IDs.
    global pivot_row
    pivot_row = 0 # Where the next pivot should go, also relevant for frontier for neat drawing.
    frontier_row = 0 # Where the frontier should go only according to internal verts.
    global gadgets
    gadgets = [] # Needed later, may not actually exist in graph, depth ordered.
    while True:
        # Neater drawing
        # (`max(..,..)` instead of `0` breaks if we add too many pivots faster
        #  than the frontier advances, pretty rare.)
        current_frontier_row = max(pivot_row, frontier_row)
        for f in frontier:
            g.set_row(f, current_frontier_row)
        for b in backrow:
            if b in g.vertices():
                g.set_row(b, -1)
        
        if not quiet:
            print(f"Frontier: {frontier}")
        if not dontdraw:
            zx.draw(g, labels = True)
        if verify_stepwise and not zx.compare_tensors(g, circuit.to_graph()):
            raise ValueError("Equality lost with above graph!")
        # What `frontier` indices have the closest neighbours.
        # This determines what vertices we will (simultaneously) handle now.
        # (Exclude all boundary neighbours for an easy "done" check.)
        closest_frontier_indices : Set[int] = set()
        closest_row : FloatInt = float('inf')
        
        for i in range(len(frontier)):
            fi = frontier[i]
            frn = right_neighbour(g, fi)
            if frn == None:
                continue
            row = g.row(frn)
            if row < closest_row:
                closest_row = row
                closest_frontier_indices.clear()
            if row == closest_row:
                closest_frontier_indices.add(i)
        
        if closest_row == float('inf'):
            break

        # Choose an arbitrary vertex to handle this iter
        i = closest_frontier_indices.pop()
        fi = frontier[i]
        fi_qubit = g.qubit(fi)
        fni = right_neighbour_raise_if_none(g, fi)
        fnni = right_neighbour(g, fni)
        old_fi = None

        if not quiet:
            print(f"Chose frontier[{i}] = {fi} this iteration.")

        # We need to do two things:
        # 1: Handle the new vert `fni`:
        # 1a) while `(fi,fni)`` is `SIMPLE`, merge the verts;
        # 1b) otherwise, do nothing (step 2 messes with this H-edge).
        # 2: Postprocess the resulting frontier:
        # 2a) if `fi` ends up with lcomp'able phase, lcomp;
        # 2b) if it otherwise has pivotable phase, pivot it and another vert
        #     at or to the left of the frontier (if any available);
        # 2c) if it otherwise has no pivotable phase but another vert at or
        #     to the left of the frontier has, apply that pivot instead.
        # 2d) update the frontier depending on the (re)moves of pivot/lcomp.
        #     pyzx' lcomp/pivot gives no guarantees about what gets removed,
        #     so keep that in mind.

        if g.edge_type((fi, fni)) == EdgeType.SIMPLE:
            if not quiet:
                print(f"▶ Merging vertices ({fi}, {fni}) → {fi}")
            merge_vertices(g, (fi, fni))
            fni = fnni
            # The code afterwards assumes a H-edge for the pivot/lcomp.
            # So keep removing these until no simple edges.
            continue

        if has_lcomp_phase(g, fi):
            if not quiet:
                print(f"▶ LComping vertex {fi}")
            lcomp_single_vertex_raise_if_none(g, fi, f"graphliker_lcomp ({fi})", quiet)
        else:
            # If `fi` has pivot phase:
            # Find any neighbour in the past/not on or beyond the frontier.
            # Prefer non-Clifford neighbours.
            # Otherwise:
            # Find any neighbour in the past with pivotable phase. If this
            # doesn't exist, advance the frontier.
            candidate : int | None = None
            fi_row = g.row(fi)
            if has_pivot_phase(g, fi):
                for v in g.neighbors(fi):
                    if g.row(v) >= fi_row:
                        continue
                    candidate = v
                    if not has_clifford_phase(g, v):
                        break
                if candidate == None:
                    raise ValueError(f"Weird branch with frontier vert {fi} (index {i}) (did you forget to move back a vertex that is not longer a frontier?)")
            else:
                for v in g.neighbors(fi):
                    if g.row(v) >= fi_row:
                        continue
                    if has_pivot_phase(g, v):
                        candidate = v
                        break
                if candidate == None:
                    # This vertex is actually hopeless, a non-Clifford connected to
                    # only non-Cliffords in the GSLC. Move the frontier ahead.
                    # (In case it's not obvious, no pivot happens this branch.)
                    frontier_row = current_frontier_row + 1
                    frontier[i] = fni
                    if not quiet:
                        print(f"▶ Updating frontier[{i}] (double NC): {fi} → {fni}")
                    continue
            e = (candidate, fi)
            if not quiet:
                print(f"▶ Pivoting edge {e}")
            pivot_single_edge_raise_if_none(g, e, f"graphliker_pivot {e}", quiet)
            # The branch below works fine when `fi` is a removed vertex.
            # But if `candidate` is a removed vertex, we're in trouble.
            # In that case, shift everything over.
            if not candidate in g.vertices():
                frontier[i] = fni
                if not quiet:
                    print(f"▶ Updating frontier[{i}] (removed backrow): {fi} → {fni}")
                old_fi = fi
                fi = fni
                fni = fnni
                fnni = None

        # `fi` is no longer a frontier if it gets deleted or pushed to be a gadget.
        # In that case, shift the frontier over.
        # TODO: It can no longer be pushed to be a gadget, right?
        if not fi in g.vertices() or g.qubit(fi) != fi_qubit:
            if fni == None:
                raise ValueError(f"Weird branch #2 with frontier vert {fi} (index {i})'s right neighbour being None after merge or backrow removal")
            # The pivot doesn't touch vertex fni, it only modifies connectivity
            # from before fni up to fni.
            frontier[i] = fni
            if not quiet:
                print(f"▶ Updating frontier[{i}]: {fi} → {fni}")
        
        # It may instead also happen that fi's row is the frontier's row while
        # no longer being a frontier vertex. In that case, move it back:
        # - If fi is adjacent to the left boundary, move it neatly to the LHS;
        # - Otherwise, move it back by 1 (arbitrary >0).
        if old_fi != None and old_fi in g.vertices() and g.row(old_fi) == current_frontier_row:
            b = get_adjacent_boundary(g, old_fi)
            if b == None or g.row(b) > frontier_row:
                g.set_row(old_fi, frontier_row - 1)
            else:
                backrow.add(old_fi)
                g.set_qubit(old_fi, g.qubit(b))

    if not dontdraw:
        zx.draw(g, labels=True)

    # I am sad. This is instead of the commented stuff below.
    print(f"▶ Doing standard pyzx.full_reduce on smaller graph.")
    zx.full_reduce(g, quiet=quiet)

    if not dontdraw:
        zx.draw(g, labels=True)

    # # Walk through each gadget's ought-to-be-but-may-not-be phaseless spider,
    # # and see if it's connected with an input/output that it can be pivoted with.
    # # Otherwise, it may be pivoted with another gadget's phaseless sometimes.
    # for (v,_) in list(gadgets):
    #     # pyzx' pivot does not define which vertex gets removed; this one may
    #     # be able to be pivoted multiple times.
    #     while v in g.vertices() and has_pivot_phase(g,v) and not adjacent_to_boundary(g, v):
    #         pivot_neighs = get_all_adjacent_where(g, v, lambda g,v,n: has_pivot_phase(g, n))
    #         if len(pivot_neighs) == 0:
    #             break
    #         boundary_adjacent_pivot_neighs = [n for n in pivot_neighs if adjacent_to_boundary(g, n)]
    #         w = boundary_adjacent_pivot_neighs[0] if len(boundary_adjacent_pivot_neighs) > 0 else pivot_neighs[0]
    #         print(f"▶ Cleanup pivot ({v},{w})")
    #         pivot_single_edge_raise_if_none(g, (v,w), "gadget cleanup pivot", quiet=quiet)

    #         if not dontdraw:
    #             zx.draw(g, labels=True)
    #         if verify_stepwise and not zx.compare_tensors(g, circuit.to_graph()):
    #             raise ValueError("Equality lost with above graph!")

    return g

def lcomp_single_vertex_raise_if_none(g : Graph, u : int, name : str, quiet : bool):
    # Add dummy spiders if not internal
    for b in list(g.neighbors(u)):
        if g.type(b) == VertexType.BOUNDARY:
            # =u -- v -- b
            v = insert_vertex(g, (u,b), VertexType.Z, 0, move_vertices = False, extra_hadamards = False)
            if g.edge_type((u,v)) != EdgeType.HADAMARD:
                flip_edge_type(g, (u,v))
                flip_edge_type(g, (v,b))

    if lcomp_single_vertex(g, u, name, quiet) == 0:
        print("Problem graph:")
        zx.draw(g, labels = True)
        raise ValueError(f"Expected to lcomp {u}, but did not!")

def pivot_single_edge_raise_if_none(g : Graph, e : Tuple[int,int], name : str, quiet : bool):
    if pivot_single_edge_boundary(g, e, name, quiet) == 0:
        print("Problem graph:")
        zx.draw(g, labels = True)
        raise ValueError(f"Expected to pivot {e}, but did not!")

def right_neighbour_raise_if_none(g : Graph, v : int) -> int:
    """ The same as `right_neighbour` but throws if it would return `None`."""
    n = right_neighbour(g, v)
    if n == None:
        print("Problem graph:")
        zx.draw(g, labels = True)
        raise ValueError("The right neighbour is a BOUNDARY!")
    return n

def right_neighbour(g : Graph, v : int) -> int | None:
    """ Returns the right neighbor of a vertex, or None if it's a BOUNDARY.

    This right neighbour is the neighbour that lies on the same qubit
    immediately to the right.
    """
    if g.type(v) == VertexType.BOUNDARY:
        raise ValueError(f"BOUNDARY vertex {v} is not valid input.")

    v_row = g.row(v)
    v_qubit = g.qubit(v)
    right = [u for u in g.neighbors(v) if g.row(u) > v_row and g.qubit(u) == v_qubit]
    if len(right) != 1:
        print("Problem graph:")
        zx.draw(g, labels = True)
        raise ValueError(f"Vertex {v} doesn't have exactly one right neighbour on qubit {v_qubit} (instead {right}), which is not allowed.")
    right = right[0]
    if g.type(right) == VertexType.BOUNDARY:
        return None
    return right

def pivot_single_edge_boundary(g : Graph, e : Tuple[int,int], name : str, quiet : bool) -> int:
    """ Slightly generalised pyzx pivot that supports phases as well
    (resulting in phase gadgets), and also sees regular edges as BOUNDARY.
    """
    (u,v) = e
    u_boundary_neighs = {w for w in g.neighbors(u) if g.type(w) == VertexType.BOUNDARY or g.edge_type((u,w)) == EdgeType.SIMPLE}
    v_boundary_neighs = {w for w in g.neighbors(v) if g.type(w) == VertexType.BOUNDARY or g.edge_type((v,w)) == EdgeType.SIMPLE}
    u_phase = g.phase(u)
    v_phase = g.phase(v)
    u_neighs = set(g.neighbors(u)).difference({v}).difference(u_boundary_neighs)
    v_neighs = set(g.neighbors(v)).difference({u}).difference(v_boundary_neighs)
    uv_neighs = u_neighs.intersection(v_neighs)
    u_neighs.difference_update(uv_neighs)
    v_neighs.difference_update(uv_neighs)

    # Delegate to pyzx' pivot after turning non-pivotable phases of u/v into
    # gadgets above. These do not have proper positions if they exist, yet.
    global gadgets
    gadget_u : Tuple[int,int] | None = None
    gadget_v : Tuple[int,int] | None = None
    if not has_pivot_phase(g, u):
        w2 = add_vertex(g, 0, 0, VertexType.Z, u_phase)
        g.set_phase(u, 0)
        g.add_edge((u,w2), EdgeType.SIMPLE)
        w1 = insert_vertex(g, (u,w2), VertexType.Z, 0, extra_hadamards = True)
        gadget_u = (w1, w2)

        gadgets.append(gadget_u)
    if not has_pivot_phase(g, v):
        w2 = add_vertex(g, 0, 0, VertexType.Z, v_phase)
        g.set_phase(v, 0)
        g.add_edge((v,w2), EdgeType.SIMPLE)
        w1 = insert_vertex(g, (v,w2), VertexType.Z, 0, extra_hadamards = True)
        gadget_v = (w1, w2)

        gadgets.append(gadget_v)

    # PyZX's pivot also considers only BOUNDARY vertices a boundary, instead
    # of any SIMPLE edge. Temporary hack: make everything a boundary.
    boundarified = {w for w in u_boundary_neighs if not g.type(w) == VertexType.BOUNDARY}
    boundarified.update(w for w in v_boundary_neighs if not g.type(w) == VertexType.BOUNDARY)
    for w in boundarified:
        g.set_type(w, VertexType.BOUNDARY)

    # Now manually construct the single match we care about.
    # To quote the pyzx documentation:
    # Perform a pivoting rewrite, given a list of matches as returned by
    # ``match_pivot(_parallel)``. A match is itself a list where:
    #
    # ``m[0]`` : first vertex in pivot.
    # ``m[1]`` : second vertex in pivot.
    # ``m[2]`` : list of zero or one boundaries adjacent to ``m[0]``.
    # ``m[3]`` : list of zero or one boundaries adjacent to ``m[1]``.
    if len(u_boundary_neighs) > 1:
        raise ValueError(f"{u} being connected to *multiple* boundaries {u_boundary_neighs} is not supported.")
    if len(v_boundary_neighs) > 1:
        raise ValueError(f"{v} being connected to *multiple* boundaries {v_boundary_neighs} is nots upported.")
    match = (
        u,
        v,
        list(u_boundary_neighs),
        list(v_boundary_neighs)
    )

    matched = False
    def match_match_once(g = None, matchf = None):
        nonlocal matched
        if not matched:
            matched = True
            return [match]
        return []

    iters = simplify.simp(
        g,
        name,
        match_match_once,
        simplify.pivot,
        quiet=quiet
    )

    # Undo the boundary hack from before
    for w in boundarified:
        g.set_type(w, VertexType.Z)

    # Note: If the gadget has Clifford phase, it can be simplified.
    # Phases 0,π result in removal of the entire gadget, with the gadget phase
    # added to the gadget's neighbours (v_neighs, uv_neighs).
    # Phases ±π/2 do the same, but also adds edges between all of v_neighs,
    # uv_neighs (both between classes and within).
    for (gadget, neighbours) in [(gadget_u, v_neighs.union(uv_neighs)), (gadget_v, u_neighs.union(uv_neighs))]:
        if gadget == None:
            continue
        (w1,w2) = gadget
        # Non-Clifford gadgets stay. Put them in a neat place.
        if not has_clifford_phase(g, w2):
            global pivot_row
            g.set_position(w1, -1, pivot_row)
            g.set_position(w2, -2, pivot_row)
            pivot_row += 1
            continue
        # Other gadgets get removed.
        w2_phase = g.phase(w2)
        for n in neighbours:
            n_phase = g.phase(n) + w2_phase
            g.set_phase(n, n_phase)
        if w2_phase == Fraction(1,2) or w2_phase == Fraction(3,2):
            for n1 in neighbours:
                for n2 in neighbours:
                    if n1 <= n2:
                        continue
                    g.add_edge_smart((n1, n2), edgetype = EdgeType.HADAMARD)
        g.remove_vertices(gadget)
        # bleh
        if gadget == gadget_u:
            gadget_u = None
        if gadget == gadget_v:
            gadget_v = None
    
    # Note II: If the gadget has arity 1, it can be removed no matter the phase.
    for gadget in [gadget_u, gadget_v]:
        if gadget == None:
            continue
        (w1,w2) = gadget
        neighs = get_all_adjacent_where(g, w1, lambda g,v,n: n != w2)
        if len(neighs) == 1:
            w0 = neighs[0]
            w0_phase = g.phase(w0) + g.phase(w2)
            g.set_phase(w0, w0_phase)
            g.remove_vertices(gadget)
    
    return iters

def adjacent_to_pivot_boundary(g : Graph, v : int) -> bool:
    """ Whether the pivot sees a pivot vertex as boundary. """
    if is_adjacent_to_boundary(g, v):
        return True
    for w in g.neighbors(v):
        if g.edge_type((v,w)) == EdgeType.SIMPLE:
            return True
    return False

if __name__ == '__main__':
    circ = Circuit(2)
    circ.add_gate(zx.gates.ZPhase(0, Fraction(1,4)))
    circ.add_gate(zx.gates.CNOT(0, 1))
    circ.add_gate(zx.gates.CNOT(0, 1))
    g = to_graphlike(circ)