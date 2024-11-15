from copy import deepcopy
import random
from extract_lcomp import *
import pyzx as zx
import time

# Output:
# [pyzx] [custom] [time]   [Name]
#    557        ~   1.4x   random circuit #1 (greed 1)
#    557      -5%   1.7x   random circuit #1 (greed 2)
#    557      -9%   2.0x   random circuit #1 (greed 3)
#    557     -12%   2.7x   random circuit #1 (greed 4)
#    557     -10%   3.5x   random circuit #1 (greed 5)
#    557      -9%   5.1x   random circuit #1 (greed 6)
#    557     -15%   6.3x   random circuit #1 (greed 7)
#    557     -11%   7.5x   random circuit #1 (greed 8)
#    557      -9%   6.7x   random circuit #1 (greed 9)
#
#    442        ~   1.7x   random circuit #2 (greed 1)
#    442     -13%   1.9x   random circuit #2 (greed 2)
#    442     -19%   2.5x   random circuit #2 (greed 3)
#    442     -17%   3.2x   random circuit #2 (greed 4)
#    442     -18%   4.7x   random circuit #2 (greed 5)
#    442     -18%   7.2x   random circuit #2 (greed 6)
#    442     -18%   9.0x   random circuit #2 (greed 7)
#    442     -18%  10.2x   random circuit #2 (greed 8)
#    442     -18%  10.9x   random circuit #2 (greed 9)
#
#    460        ~   1.7x   random circuit #3 (greed 1)
#    460     -13%   2.1x   random circuit #3 (greed 2)
#    460     -16%   2.2x   random circuit #3 (greed 3)
#    460     -14%   2.9x   random circuit #3 (greed 4)
#    460     -14%   4.2x   random circuit #3 (greed 5)
#    460     -12%   6.6x   random circuit #3 (greed 6)
#    460     -11%   7.5x   random circuit #3 (greed 7)
#    460     -11%   8.4x   random circuit #3 (greed 8)
#    460     -11%   9.6x   random circuit #3 (greed 9)
#
#    545        ~   1.6x   random circuit #4 (greed 1)
#    545      -9%   1.9x   random circuit #4 (greed 2)
#    545     -11%   2.0x   random circuit #4 (greed 3)
#    545     -14%   3.1x   random circuit #4 (greed 4)
#    545     -15%   4.3x   random circuit #4 (greed 5)
#    545     -13%   5.6x   random circuit #4 (greed 6)
#    545     -11%   7.5x   random circuit #4 (greed 7)
#    545     -14%   8.2x   random circuit #4 (greed 8)
#    545     -14%   8.9x   random circuit #4 (greed 9)

# funny hoe op random circuits 't beter werkt en de runtime lin lijkt in greed
# meanwhile op actual circuits is 't 10% shitter en zie ik exp gedrag

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
    
    print_str = f"{factor_2qubit:+7}% {factor_time:5.1f}x   {name} (greed {greed})"
    print_str = print_str.replace("+0%", "  ~")
    print(print_str)
    exit()

if __name__ == '__main__':
    import multiprocessing
    import sys
    timeout : float = 300

    random.seed(1)
    qubits = 10
    gates = 1000
    α = 0.5
    circuits = [
        zx.generate.CNOT_HAD_PHASE_circuit(qubits, gates, p_had = 0.2, p_t = 0.2),
        zx.generate.CNOT_HAD_PHASE_circuit(qubits, gates, p_had = 0.6, p_t = 0.2),
        zx.generate.CNOT_HAD_PHASE_circuit(qubits, gates, p_had = 0.2, p_t = 0.6),
        zx.generate.CNOT_HAD_PHASE_circuit(qubits, gates, p_had = 0.33, p_t = 0.33)
    ]

    print_header()
    for i in range(len(circuits)):
        for greed in range(1,10):
            try:
                name = f"random circuit #{i+1}"
                g = circuits[i].to_graph()
                zx.simplify.full_reduce(g, quiet = True)

                p = multiprocessing.Process(target=test_circuit, args=(g, name, greed, α))
                p.start()
                p.join(timeout)
                sys.stdout.flush()
                if p.is_alive():
                    print(f"                         {name} timed out after {timeout} seconds")
                    p.terminate()
                    p.join()
            except:
                pass
        print("")