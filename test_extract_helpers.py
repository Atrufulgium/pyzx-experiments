from typing import Dict, Tuple
from extract_helpers import *
import os
import pyzx as zx
import unittest

# This file depends on the existence of a few files (as I'm not going to embed
# graph creation in this file):
# - there_back_ex_2_43.qgraph               (Example 2.43 in There and Back Again)
# - unfused.qgraph                          (Single node with two XY(π) measurements attached)
# - mod5_4_before_circuit_graph.qgraph      (Large example in AllFeatures.ipynb, circuity)
# - mod5_4_before_optimised_graph.qgraph    (Large example in AllFeatures.ipynb, H-edge'y)
# - no_gflow.qgraph                         (Small diagram without gflow, as in Remark 2.38 in There and Back Again)

# Note that unittests are probably found by reflection by getting all classes
# implementing a unittest.XXX class. Then it apparantly looks for all
# methods test_XXX(self) to actually run.

def open_qgraph(name : str) -> Graph:
    """ Opens ./name.graph as a graph."""
    with open(os.path.join(f'./{name}.qgraph'), 'r') as f:
        g = zx.Graph.from_json(f.read())
    return g

def open_qgraph_open_graph(name : str) -> OpenGraph:
    return OpenGraph(open_qgraph(name))

class TestMeasurements(unittest.TestCase):
    
    def test_three_planes(self):
        self.compare_from_qgraph_file("there_back_ex_2_43", [
            Measurement('XY', Fraction(1,23)),
            Measurement('XY', Fraction(2,23)),
            Measurement('XZ', Fraction(3,23)),
            Measurement('YZ', Fraction(4,23))
        ])
    
    def test_unfused_error(self):
        with self.assertRaisesRegex(ValueError, ".*The given vertex can be interpreted as measurement in multiple ways.*"):
            self.compare_from_qgraph_file("unfused", [])
    
    def test_ignore_unfused_error(self):
        self.compare_from_qgraph_file("unfused", [
            Measurement('XY', 1),
            Measurement('XY', 1)
        ], throw_on_multiple=False)

    def test_large_XY_only(self):
        g = open_qgraph("mod5_4_before_circuit_graph")
        # im not manually writing out these 28 measurements
        self.compare_measurements(
            g,
            [Measurement('XY', f) for f in [g.phase(v) for v in g.vertices()] if f == Fraction(1,4) or f == Fraction(7,4)]
        )
    
    def test_large_XY_only_2(self):
        # just read this from the AllFeatures.ipynb result
        # note that they're all phase gadgets => YZ measurements
        # note in particular here there are a *lot* of 0-measurements XY's excluded
        self.compare_from_qgraph_file("mod5_4_before_optimised_graph", [
            Measurement('YZ', Fraction(1,4)),
            Measurement('YZ', Fraction(1,4)),
            Measurement('YZ', Fraction(1,4)),
            Measurement('YZ', Fraction(1,4)),
            Measurement('YZ', Fraction(7,4)),
            Measurement('YZ', Fraction(7,4)),
            Measurement('YZ', Fraction(7,4)),
            Measurement('YZ', Fraction(7,4))
        ])
    
    def compare_from_qgraph_file(self, filename : str, expected : List[Measurement], throw_on_multiple : bool = True):
        g = open_qgraph(filename)
        self.compare_measurements(g, expected, throw_on_multiple)

    def compare_measurements(self, g : Graph, expected : List[Measurement], throw_on_multiple : bool = True):
        """Compares the measurements of a graph with a list up to some data.
        
        In particular, this creates a measurement list from `g` and compares it
        with the `expected` list, but:
        - Order is ignored;
        - `None` is ignored;
        - The vertices stored within each Measurement is ignored.
        
        Only the planes and the angles need to be correct."""
        actual : List[Measurement] = []
        out : List[Measurement] = []
        for v in g.vertices():
            if (try_get_measurement(g, v, out, throw_on_multiple)):
                for o in out:
                    actual.append(o)
        
        # These dicts store the two variables of measurement we care about and
        # the number of instances of each.
        actual_dict : Dict[Tuple[Literal['XY', 'XZ', 'YZ'], FractionLike], int] = {}
        expected_dict : Dict[Tuple[Literal['XY', 'XZ', 'YZ'], FractionLike], int] = {}

        for m in actual:
            if m.plane == None:
                continue
            self.incr_dict_counter(actual_dict, (m.plane, m.phase))
        for m in expected:
            if m.plane == None:
                continue
            self.incr_dict_counter(expected_dict, (m.plane, m.phase))

        self.assertEqual(actual_dict, expected_dict, f"Full measurements dicts (ignore expected's verts):\n  Actual: {actual}\n  Expected: {expected}")
    
    def incr_dict_counter(
        self,
        dict : Dict[Tuple[Literal['XY', 'XZ', 'YZ'], FractionLike], int],
        key : Tuple[Literal['XY', 'XZ', 'YZ'], FractionLike]
    ):
        if (key in dict.keys()):
            dict[key] = dict[key] + 1
        else:
            dict[key] = 1

class TestMaximallyDelayedGFlow(unittest.TestCase):

    def test_three_planes(self):
        graph = open_qgraph_open_graph("there_back_ex_2_43")
        out = find_max_delayed_flow(graph)
        self.assertTrue(out)
        if out:
            (g, d) = out
            self.assertTrue(verify_gflow(graph, g, d))
    
    def test_gflow_large(self):
        graph = open_qgraph_open_graph("mod5_4_before_circuit_graph")
        out = find_max_delayed_flow(graph)
        self.assertTrue(out)
        if out:
            (g, d) = out
            self.assertTrue(verify_gflow(graph, g, d))

    def test_gflow_large_phase_gadgets(self):
        graph = open_qgraph_open_graph("mod5_4_before_optimised_graph")
        out = find_max_delayed_flow(graph)
        self.assertTrue(out)
        if out:
            (g, d) = out
            self.assertTrue(verify_gflow(graph, g, d))
    
    def test_no_gflow(self):
        graph = open_qgraph_open_graph("no_gflow")
        out = find_max_delayed_flow(graph)
        self.assertFalse(out)

if __name__ == '__main__':
    unittest.main()