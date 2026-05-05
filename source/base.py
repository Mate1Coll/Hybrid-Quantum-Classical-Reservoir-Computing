"""Base utilities for reproducible quantum-classical reservoir experiments.

This module defines the base class used throughout the repository to
control random number generation and enforce deterministic behavior.
It also configures numerical libraries to use a single thread in order
to avoid nondeterministic parallel behavior during batch experiments.
"""

from os import environ

# We set the number of threads to one in order to outperform parallelization
environ["OMP_NUM_THREADS"] = "1"
environ["OPENBLAS_NUM_THREADS"] = "1"
environ["MKL_NUM_THREADS"] = "1"
environ["VECLIB_MAXIMUM_THREADS"] = "1"
environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

class BaseSeededClass:

    def __init__(self, seed=None, **kwargs):
        
        """Initialize the seeded base class.

        Parameters
        ----------
        seed : int or None, optional
            Optional random seed for reproducibility. If ``None``, a random
            seed is generated internally using NumPy's default RNG.
        **kwargs
            Additional keyword arguments are accepted for subclass compatibility.

        Notes
        -----
        This constructor sets up a `numpy.random.Generator` instance as
        ``self.rng`` and stores the selected seed in ``self.seed``.
        The single-threaded environment variable settings help ensure that
        computational backends like OpenBLAS, MKL, and NumExpr do not use
        parallel threads, which can reduce variability across runs.
        """

        self.seed = seed if seed is not None else np.random.randint(1e9)  # Set the seed for reproducibility, if none, it is set randomly
        self.rng = np.random.default_rng(self.seed)  # Set the random number generator with the seed
        # print(f"[BaseSeededClass] Seed: {self.seed}")

        super().__init__()  # required for inheritance