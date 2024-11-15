from copy import deepcopy
from extract_lcomp import *
import os
import pyzx as zx
import time
from glob import glob

# Output (excl annotations):
# CNOTs in resulting circuit (max greed 4)
# [pyzx] [custom] [time]   [Name]
#    366        ~   2.2x   adder_8.qasm
#    199     -11%   1.6x   barenco_tof_10.qasm
#     21        ~   1.4x   barenco_tof_3.qasm
#     56      -2%   1.7x   barenco_tof_4.qasm
#     71      -2%   2.0x   barenco_tof_5.qasm
#    161     -12%   2.0x   csla_mux_3.qasm
#    296      -1%   2.5x   csum_mux_9.qasm
#   2790      -2%   2.0x   gf2^10_mult.qasm
#   7898        ~   2.4x   gf2^16_mult.qasm
#    304      -6%   1.5x   gf2^4_mult.qasm
#    432        ~   1.7x   gf2^5_mult.qasm
#    811      -1%   2.0x   gf2^6_mult.qasm
#   1195      -2%   1.9x   gf2^7_mult.qasm
#   1691      -2%   2.0x   gf2^8_mult.qasm
#   1977      -3%   1.8x   gf2^9_mult.qasm
#    259      +4%   1.9x   grover_5.qasm
#   2187      -2%   1.4x   ham15-high.qasm
#    333      +1%   2.0x   ham15-low.qasm       !!! worsened
#    464      -1%   1.8x   ham15-med.qasm
#    138      -5%   1.8x   hwb6.qasm
#     23        ~   1.2x   mod5_4.qasm
#     88     -15%   1.6x   mod_mult_55.qasm
#    149        ~   1.9x   mod_red_21.qasm
#    366      -1%   2.1x   qcla_adder_10.qasm
#    222      +1%   1.7x   qcla_com_7.qasm      !!! worsened
#    674      -7%   1.1x   qcla_mod_7.qasm
#     64      -2%   1.8x   qft_4.qasm
#    115     -12%   2.2x   rc_adder_6.qasm
#    133      -6%   2.0x   tof_10.qasm
#     25        ~   1.4x   tof_3.qasm
#     42        ~   1.5x   tof_4.qasm
#     54      -6%   1.8x   tof_5.qasm
#     74      -3%   1.9x   vbe_adder_3.qasm

# Vergeleken met die eerdere simulated annealing vid:
# [pyzx] [dit] [dat]
#    366     ~   -9%       adder_8.qasm
#    199  -11%  -18%       barenco_tof_10.qasm
#     21     ~   -5%       barenco_tof_3.qasm
#     56   -2%  -32%       barenco_tof_4.qasm
#     71   -2%  -19%       barenco_tof_5.qasm
#    161  -12%  -30%       csla_mux_3.qasm
#    296   -1%   -8%       csum_mux_9.qasm
#   2790   -2%  -26%       gf2^10_mult.qasm
#    304   -6%  -31%       gf2^4_mult.qasm
#    432     ~  -46%       gf2^5_mult.qasm
#    811   -1%  -30%       gf2^6_mult.qasm
#   1195   -2%  -23%       gf2^7_mult.qasm
#   1691   -2%  -25%       gf2^8_mult.qasm
#   1977   -3%  -20%       gf2^9_mult.qasm
#    259   +4%   -2%       grover_5.qasm
#    333   +1%  -15%       ham15-low.qasm
#    138   -5%  -12%       hwb6.qasm
#     23     ~  -39%       mod5_4.qasm
#     88  -15%  -11%       mod_mult_55.qasm     !!! wow de enige die ik win
#    149     ~  -30%       mod_red_21.qasm
#    366   -1%  -22%       qcla_adder_10.qasm
#    222   +1%  -18%       qcla_com_7.qasm
#    674   -7%  -29%       qcla_mod_7.qasm
#     64   -2%  -18%       qft_4.qasm
#    115  -12%  -13%       rc_adder_6.qasm
#    133   -6%  -35%       tof_10.qasm
#     25     ~  -36%       tof_3.qasm
#     42     ~  -40%       tof_4.qasm
#     54   -6%  -28%       tof_5.qasm
#     74   -3%  -40%       vbe_adder_3.qasm
# aka trash

def print_header():
    print("[pyzx] [custom] [time]   [Name]")

def test_circuit(g : Graph, name : str, greed : int, α : float):
    """ Output format:
    ```
    [pyzx] [custom] [time]   [Name]
       123      -1%   1.2x   my_circ.qc
    ```"""

    global extra_params
    extra_params["max_greed"] = greed
    extra_params["target_preference"] = α

    g_copy = deepcopy(g)
    t_start = time.perf_counter()
    out = zx.extract_circuit(g_copy)
    t_end = time.perf_counter()
    out = zx.optimize.basic_optimization(out.to_basic_gates(), do_swaps = True).to_basic_gates()
    pyzx_2qubits = out.twoqubitcount()
    pyzx_time = t_end - t_start

    g_copy = deepcopy(g)
    t_start = time.perf_counter()
    out = extract_circuit(g_copy)
    t_end = time.perf_counter()
    out = zx.optimize.basic_optimization(out.to_basic_gates(), do_swaps = True).to_basic_gates()
    custom_2qubits = out.twoqubitcount()
    custom_time = t_end - t_start

    print(f"{pyzx_2qubits:6} ", end="")

    factor_2qubit = int(100 * custom_2qubits / pyzx_2qubits) - 100
    factor_time = custom_time / pyzx_time
    
    print_str = f"{factor_2qubit:+7}% {factor_time:5.1f}x   {name}"
    print_str = print_str.replace("+0%", "  ~")
    print(print_str)
    exit()

if __name__ == '__main__':
    import multiprocessing
    import sys
    timeout : float = 300

    max_greed = 4
    α = 0.5
    print(f"CNOTs in resulting circuit (max greed {max_greed}, α {α})")

    print_header()
    files = [y for x in os.walk("./circuits/") for y in glob(os.path.join(x[0], '*.qasm'))]
    for f in files:
        try:
            name = os.path.basename(f)
            c = zx.Circuit.load(f)
            g = c.to_graph()
            zx.simplify.full_reduce(g, quiet = True)

            p = multiprocessing.Process(target=test_circuit, args=(g, name, max_greed, α))
            p.start()
            p.join(timeout)
            sys.stdout.flush()
            if p.is_alive():
                print(f"                         {name} timed out after {timeout} seconds")
                p.terminate()
                p.join()
        except:
            pass
