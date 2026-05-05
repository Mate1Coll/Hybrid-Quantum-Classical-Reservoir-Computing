"""Compute energy gap ratios for Hamiltonian disorder parameter sweeps.

This script evaluates spectral gap ratio statistics for disordered quantum
Hamiltonians as a function of field strength and disorder amplitude.
"""

import context
import numpy as np
from source.hamiltonian import Ergodicity

if __name__ == "__main__":

    L = 9 # number of qubits
    Js = 1
    n_h = 30
    n_W = 30
    z_block = 1

    alpha = 0
    n_iter = 1000

    seed = 23

    store = True
    mean = True

    # Specific h,W values for distribution:
    h_val = np.logspace(-2,2, num=20)
    W_val = np.logspace(-2,2, num=20)

    for h,W in zip(h_val, W_val):
        Ergodicity.rn_distribution(L, Js, h, W, z_block, alpha, n_iter, seed, store=store)

    if mean:
        start = context.time.time()
        Ergodicity.gap_ratio_W_h(L, Js, n_h, n_W, z_block, alpha, n_iter, seed, store=store, parallel=True)
        end = context.time.time()
        print(f"Time taken: {end - start} seconds {(end - start) / 60} minutes, {(end - start) / 3600} hours")
        print("Finished")