from cgi import test
from copy import deepcopy
from lcomping import *
from pyzx.circuit import Circuit
import os
from glob import glob

def print_header():
    print("[pyzx] [Shallow] [Deep] [Few] [Many] [Maintain]   [Name]")

def test_circuit(c : Circuit, name : str):
    """ Output format:
    ```
    [pyzx] [Shallow] [Deep] [Few] [Many] [Maintain]   [Name]
       123       -1%   +10%   -0%    +5%      -230%   my_circ.qc
    ```"""
    g = cast(Graph, c.to_graph())
    
    strategies = [
        (zx.simplify.lcomp_simp, zx.simplify.pivot_simp),
        (lcomp_simp_shallowest_first, pivot_simp_shallowest_first),
        (lcomp_simp_deepest_first, pivot_simp_deepest_first),
        (lcomp_simp_few_edges, pivot_simp_few_edges),
        (lcomp_simp_many_edges, pivot_simp_many_edges),
        (lcomp_simp_maintain_edges, pivot_simp_maintain_edges)
    ]
    vert_counts = []
    edge_counts = []
    for (lcomp, pivot) in strategies:
        gclone = deepcopy(g)
        interior_clifford_simp_alt(gclone, quiet=True, lcomp_map=lcomp, pivot_map=pivot)
        vert_count = gclone.num_vertices()
        edge_count = gclone.num_edges()
        vert_counts.append(vert_count)
        edge_counts.append(edge_count)
        
        # Verify "actually done"
        interior_clifford_simp_alt(gclone, quiet=True)
        if gclone.num_edges() != edge_count or gclone.num_vertices() != vert_count:
            print(f"Was not actually done with {name} with strategy ({lcomp},{pivot})")
    
    reference = vert_counts[0]
    del vert_counts[0]
    p = []
    for i in range(5):
        count = vert_counts[i]
        p.append(int(100 * count / reference) - 100)
    print_str = f"{reference:6} {p[0]:+8}% {p[1]:+5}% {p[2]:+4}% {p[3]:+5}% {p[4]:+9}%   {name}"
    print_str = print_str.replace("+0%", "  ~")
    print(print_str)
    exit()

if __name__ == '__main__':
    import multiprocessing
    import sys
    timeout : float = 300

    print_header()
    files = [y for x in os.walk("./circuits/") for y in glob(os.path.join(x[0], '*.qc'))]
    do = ["adder_8_tpar.qc", "csla_mux_3_tpar.qc", "csum_mux_9_tpar.qc", "gf2^10_mult_tpar.qc", "gf2^16_mult_tpar.qc", "gf2^32_mult_tpar.qc", "gf2^6_mult_tpar.qc", "gf2^7_mult_tpar.qc", "gf2^8_mult_tpar.qc", "gf2^9_mult_tpar.qc", "mod_adder_1024_tpar.qc", "mod_adder_1048576_tpar.qc", "mod_red_21_tpar.qc", "qcla_adder_10_tpar.qc", "qcla_com_7_tpar.qc", "qcla_mod_7_tpar.qc", "rc_adder_6_tpar.qc", "gf2^128_mult.qc", "gf2^16_mult.qc", "gf2^256_mult.qc", "gf2^32_mult.qc", "gf2^64_mult.qc", "hwb10.qc", "hwb11.qc", "hwb12.qc", "hwb8.qc", "mod_adder_1048576.qc", "adder_8.qc", "cycle_17_3.qc", "gf2^8_mult.qc", "grover_5.qc", "ham15-high.qc", "ham15-med.qc", "mod_adder_1024.qc", "Adder16_pyzx.qc", "Adder32_pyzx.qc", "Adder64_pyzx.qc", "gf2^10_mult_pyzx.qc", "gf2^16_mult_pyzx.qc", "gf2^8_mult_pyzx.qc", "gf2^8_mult_pyzxtodd.qc", "ham15-high_pyzx.qc", "ham15-med_pyzx.qc", "hwb8_pyzx.qc", "mod_adder_1024_pyzx.qc", "nth_prime8_pyzx.qc"]
    for f in files:
        try:
            name = os.path.basename(f)
            if name not in do:
                continue
            c = zx.Circuit.load(f)
            p = multiprocessing.Process(target=test_circuit, args=(c, name))
            p.start()
            p.join(timeout)
            sys.stdout.flush()
            if p.is_alive():
                print(f"                                                  {name} timed out after {timeout} seconds")
                p.terminate()
                p.join()
        except:
            pass
