"""Echo State Network implementation for classical and quantum reservoir readouts.

This module defines the ESN classes used to process classical and quantum-derived
signals. The ESN is responsible for generating reservoir states, applying a
nonlinear activation, and producing readout features for subsequent linear
regression tasks.
"""

from .base import *
from .tasks import Tasks
import itertools
import pathlib
import os
from .utils import input_full_esn, ax_to_str, load_state_data

class Esn(BaseSeededClass):

    """Base Echo State Network model.

    The ESN is a recurrent reservoir of randomly connected neurons.
    Internal weights are initialized randomly and scaled to a unit spectral
    radius before processing inputs.
    """

    def __init__(self, N_esn, g, l, d=1, func='sigmoid', N_rep=1, **kwargs):

        """Initialize the ESN.

        Parameters
        ----------
        N_esn : int
            Number of reservoir neurons.
        g : float
            Recurrent feedback gain for the reservoir weight matrix.
        l : float
            Input scaling factor for reservoir input connections.
        d : int, optional
            Dimension of the input signal, by default 1.
        func : {'sigmoid', 'tanh'}, optional
            Activation function for reservoir neuron updates.
        N_rep : int, optional
            Number of repeated reservoir update steps per input timestep.
        **kwargs
            Extra keyword arguments passed to the base seeded class.
        """

        super().__init__(**kwargs) # initialize the parent class

        if func not in ['tanh', 'sigmoid']:
            raise ValueError("The activation function must be either 'tanh' or 'sigmoid'.")

        self.N_esn = N_esn
        self.g = g
        self.l = l
        self.func = func
        self.N_rep = N_rep

        self.d = d

    def __repr__(self):

        """ Return a representation of the Esn class. """

        return f"Esn(N_esn={self.N_esn}, g={self.g}, l = {self.l}, d={self.d}, func={self.func}, seed={self.seed})"
    
    def __str__(self):

        """Return a description of the Esn class. """

        return f"Echo State Networ with {self.N_esn} neurons, g = {self.g} feedback gain, l = {self.l} input gain and activation function '{self.func}'."        

    def set_weigths(self):

        """Initialize ESN reservoir weights and input weights.

        The internal reservoir weight matrix is sampled uniformly from [-1, 1]
        and then scaled to unit spectral radius to enforce stable reservoir
        dynamics. The input weights and initial reservoir state are also
        initialized uniformly in [-1, 1].

        Returns
        -------
        tuple
            ``(W_esn, W_in, x0)`` where:
            - ``W_esn`` is the reservoir weight matrix,
            - ``W_in`` is the input weight matrix,
            - ``x0`` is the initial reservoir state vector.
        """

        self.W_esn = self.rng.uniform(-1, 1, size=(self.N_esn, self.N_esn)) # reservoir weights
        self.W_in = self.rng.uniform(-1, 1, size=(self.N_esn, self.d)) # input weights
        self.x0 = self.rng.uniform(-1, 1, size=(self.N_esn)) # initial state of the reservoir

        rho = np.max(np.abs(np.linalg.eigvals(self.W_esn))) # spectral radius of the reservoir weights
        self.W_esn = self.W_esn / rho # normalize the reservoir weights

        return self.W_esn, self.W_in, self.x0
    
    def activation_f(self):

        """Set the activation function used by reservoir neurons.

        The ESN supports `tanh` and `sigmoid` activations. The sigmoid
        implementation clips its input to avoid overflow.
        """

        if self.func == 'tanh':
            self.af = np.tanh

        if self.func == 'sigmoid':
            self.af = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            
        return
    

    
class EsnDynamics(Esn, Tasks):

    """Echo State Network dynamics over time.

    This class combines the ESN reservoir with task and performance
    utilities from ``Tasks``. It is used to generate reservoir states and to
    feed them into linear readouts for computing task performance.
    """

    def __init__(self, N_esn, g, l, func='sigmoid', **kwargs):

        super().__init__(N_esn=N_esn, g=g, l=l, func=func, **kwargs)
        
        self.activation_f() # set the activation function

    def __repr__(self):

        """ Return a represenation of EsnDynamics """

        return Esn.__repr__(self) + "\n" + Tasks.__repr__(self)

    def __str__(self):

        """ Returna description of EsnDynamics """

        return Esn.__str__(self) + "\n" + Tasks.__str__(self)
    
    
    def input_signal_reshape(self):
    
        """Prepare the input signal for reservoir state evolution.

        This method reshapes the input time series into a 2D array with shape
        ``(n_steps, d)`` and rescales values into the ESN input range.

        Returns
        -------
        ndarray
            Reshaped and normalized input matrix for the ESN.
        """

        if not hasattr(self, 'input_signals'):
            print('There is no input signal, generating ...')
            u = self.get_input_signal() # input signal

        else:
            u = self.input_signals

        self.u = u.reshape(-1, 1) # reshape the input signal to match the dimension of W_in
        self.u = self.u * 1 / self.max_bound_input # Reescale input to ESN such u in [0,1]

        return self.u
    
    def echo_states(self):

        """Compute reservoir state dynamics for the current input signal.

        The reservoir state is updated sequentially for every timestep according
        to the ESN update equation. The resulting state matrix is stored in
        ``self.x_out``.

        Returns
        -------
        ndarray
            Time series of reservoir states with shape ``(n_steps, N_esn)``.
        """

        self.d = self.u.shape[1]
        self.set_weigths() # set the weights of the ESN

        self.x_out = np.zeros((self.n_steps, self.N_esn))
        self.x_out[0] = self.x0 # initial state of the reservoir
        arg=np.zeros_like(self.x_out)

        for i in range(1, self.n_steps):
            # Compute the echo state
            x_prev = self.x_out[i-1]
            for _ in range(self.N_rep):
                rec_term = (self.g * (self.W_esn @ x_prev))
                in_term  = (self.l * (self.W_in @ self.u[i]))
                x_new = self.af(rec_term + in_term) # update the state of the reservoir
                arg[i] = rec_term + in_term
                x_prev = x_new

            self.x_out[i] = x_new

        return self.x_out
    
    def classical_output_weights(self):

        """Compute reservoir states for a classical input and prepare them for readout.

        This method ensures the input is reshaped, then computes the reservoir
        states through ``echo_states``.

        Returns
        -------
        ndarray
            Reservoir states computed for the current input signal.
        """
        self.input_signal_reshape()
        return self.echo_states()
    

    @staticmethod
    def esn_worker(N_esn, g, l, task_name, idx_iter, axis=['x'], qtasks=[], seed=None, store=True, inp_type='qubit'):

        """Worker function that computes ESN reservoir states for a single iteration.

        Parameters
        ----------
        N_esn : int
            Number of reservoir neurons.
        g : float
            Feedback gain.
        l : float
            Input scaling.
        task_name : str
            Task identifier, e.g. ``'Qinp'`` for quantum input tasks.
        idx_iter : int
            Iteration index used for result storage.
        axis : list[str], optional
            Observable axes to use for quantum inputs.
        qtasks : list[str], optional
            Quantum task labels used when task_name is ``'Qinp'``.
        seed : int or None, optional
            Random seed for reproducibility.
        store : bool, optional
            If True, save the computed states to disk.
        inp_type : str, optional
            Quantum input type for ``Qinp`` tasks.

        Returns
        -------
        tuple
            ``(x_out, input_signals)`` computed for this iteration.
        """

        esn = EsnDynamics(N_esn, g, l, task_name=task_name, qtasks=qtasks, n_max_delay = 0, seed=seed, inp_type=inp_type)
        esn.get_input_signal()

        if task_name == "Qinp":
            esn.u = input_full_esn(esn.input_signals, inp_type, axis=axis)
            esn.echo_states()
        else:
            esn.classical_output_weights()

        if store:

            if task_name == "Qinp":
                ax_str , _ = ax_to_str(axis, caxis=[])
                pathlib.Path(f'results/data/{task_name}/ESN/{inp_type}/ESN_Nesn{N_esn}_g{g}_l{l}_ax_{ax_str}_{inp_type}').mkdir(parents=True, exist_ok=True)
                np.savez_compressed(f'results/data/{task_name}/ESN/{inp_type}/ESN_Nesn{N_esn}_g{g}_l{l}_ax_{ax_str}_{inp_type}/Iter_{idx_iter}', states=esn.x_out, inp=esn.input_signals)

            else:
                pathlib.Path(f'results/data/{task_name}/ESN/ESN_Nesn{N_esn}_g{g}_l{l}').mkdir(parents=True, exist_ok=True)
                np.savez_compressed(f'results/data/{task_name}/ESN/ESN_Nesn{N_esn}_g{g}_l{l}/Iter_{idx_iter}', states=esn.x_out, inp=esn.u)

        return esn.x_out, esn.input_signals
    
    @staticmethod
    def esn_state_g_l(N_esn, g_sweep_val, l_sweep_val, task_name, N_iter, qtasks=[], axis=['x'],
                      seed=None, rewrite=False, store=True, inp_type='qubit'):

        """Compute and optionally store ESN states over a grid of gain values.

        Parameters
        ----------
        N_esn : int
            Number of reservoir neurons.
        g_sweep_val : array-like
            Values of feedback gain to sweep.
        l_sweep_val : array-like
            Values of input scaling to sweep.
        task_name : str
            Task identifier.
        N_iter : int
            Number of iterations to compute for each parameter pair.
        qtasks : list[str], optional
            Quantum tasks used only for ``'Qinp'``.
        axis : list[str], optional
            Observable axes for quantum tasks.
        seed : int or None, optional
            Random seed.
        rewrite : bool, optional
            If True, recompute states even when saved files already exist.
        store : bool, optional
            If True, save generated state files to disk.
        inp_type : str, optional
            Input type used for quantum input tasks.

        Returns
        -------
        list
            List of results returned by ``esn_worker`` for missing or rewritten iterations.
        """

        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, 1e9, size=N_iter)

        ax_str , _ = ax_to_str(axis=axis, caxis=[])

        if type(g_sweep_val) in [int, float]:
            g_sweep_val = [g_sweep_val]
        if type(l_sweep_val) in [int, float]:
            l_sweep_val = [l_sweep_val]

        for g in g_sweep_val:
            for l in l_sweep_val:

                if qtasks == 'Qinp':
                    dir_path = f'results/data/{task_name}/ESN/{inp_type}/ESN_Nesn{N_esn}_g{g}_l{l}_ax_{ax_str}/'
                else:
                    dir_path = f'results/data/{task_name}/ESN/ESN_Nesn{N_esn}_g{g}_l{l}/'
                missing_k = []

                if rewrite:

                    missing_k = range(N_iter)

                else:

                    for k in range(N_iter):
                        file_path = dir_path + f'Iter_{k}.npz'
                        if not os.path.exists(file_path):
                            missing_k.append(k)

                    if not missing_k:
                        print(f"All iterations already computed for g = {g}, l = {l}. Skipping.")
                        continue

                args = [
                    (N_esn, g, l, task_name, idx_iter, axis, qtasks, seeds[k], store, inp_type)
                    for k, idx_iter in enumerate(missing_k)
                ]

                results = Parallel(n_jobs=-1)(
                    delayed(EsnDynamics.esn_worker)(*arg) for arg in args
                )

        return results
    
    def esn_performance(
        N_esn, task_name, n_min_delay, n_max_delay, N_iter,
        sweep=True,  g_fixed=None, l_fixed=None, g_sweep_val=None, l_sweep_val=None,
        pm = 'Capacity', qtasks=[], axis=['x'], store=True, seed=None, load_data=False,
        inp_type='qubit'):

        """Evaluate ESN performance across delay and parameter settings.

        This method can either sweep reservoir hyperparameters ``g`` and ``l`` or
        evaluate performance for fixed parameter values over a range of delays.
        For quantum input tasks, it supports loading precomputed state files
        and uses quantum output performance metrics.

        Parameters
        ----------
        N_esn : int
            Number of reservoir neurons.
        task_name : str
            Task identifier such as ``'Qinp'``, ``'STM'``, or ``'NARMA'``.
        n_min_delay : int
            Minimum delay value to evaluate.
        n_max_delay : int
            Maximum delay value to evaluate.
        N_iter : int
            Number of independent iterations per parameter setting.
        sweep : bool, optional
            If True, sweep over ``g_sweep_val`` and ``l_sweep_val``.
        g_fixed : float or None, optional
            Fixed feedback gain when ``sweep`` is False.
        l_fixed : float or None, optional
            Fixed input scaling when ``sweep`` is False.
        g_sweep_val : array-like or None, optional
            Values of feedback gain when ``sweep`` is True.
        l_sweep_val : array-like or None, optional
            Values of input scaling when ``sweep`` is True.
        pm : str, optional
            Performance metric, such as ``'Capacity'``.
        qtasks : list[str], optional
            Quantum tasks used for ``Qinp`` performance.
        axis : list[str], optional
            Observable axes for quantum input preprocessing.
        store : bool, optional
            If True, save performance results to disk.
        seed : int or None, optional
            Random seed for reproducibility.
        load_data : bool, optional
            If True, load precomputed ESN states instead of recomputing them.
        inp_type : str, optional
            Input type for quantum tasks.

        Returns
        -------
        tuple
            Parameter values and performance arrays depending on the selected mode.
        """

        if task_name == 'PC' and n_min_delay < 1:
            n_min_delay = 1
            print('For PC task the minimum delay is 1: n_min_delay set to 1')

        delays = list(range(n_min_delay, n_max_delay))
        task = Tasks(n_max_delay=0, task_name=task_name, pm=pm, qtasks=qtasks)
        ax_str , _ = ax_to_str(axis=axis, caxis=[]); nqtasks = task.nqtasks

        if sweep:

            if g_sweep_val is None or l_sweep_val is None:
                raise ValueError("When sweep is True, g_sweep_val and l_sweep_val must be provided.")
            
            C_shape = (len(g_sweep_val), len(l_sweep_val), N_iter, nqtasks) if task_name == 'Qinp' else (len(g_sweep_val), len(l_sweep_val), N_iter)
            C_store = np.full(C_shape, fill_value=np.nan)

            for i, g in enumerate(g_sweep_val):
                for j, l in enumerate(l_sweep_val):

                    if not load_data:
                    
                        data = EsnDynamics.esn_state_g_l(
                            N_esn, g_sweep_val=[g], l_sweep_val=[l], task_name=task_name,
                            qtasks=qtasks, axis=axis, N_iter=N_iter, seed=seed,
                            rewrite=True, store=False, inp_type=inp_type)
                        
                    for it in range(N_iter):

                        if load_data:

                            data = load_state_data(N_esn, g, l, task_name, it, ax_str=ax_str, inp_type=inp_type)
                            inp = data['inp']
                            states = data['states']
                        
                        else:

                            sample = data[it]
                            states = sample[0]
                            inp = sample[1]

                        task.input_signals = inp
                        task.x_out = states
                        C_delay = 0

                        if task_name == 'Qinp':
                            task.reset_quantum_input_features()

                        for _, n_delay in enumerate(delays):

                            task.n_max_delay = n_delay
                            task.get_output_signal(reshapeflag=True)
                            C = task.performance(qflag=True)

                            C_delay += C

                        C_store[i,j,it] = C_delay

            
            C_mean = np.mean(C_store, axis=2)
            C_std = np.std(C_store, axis=2)

            if store:

                if task_name == 'Qinp':
                    
                    strqtasks = task.strqtasks
                    pathlib.Path(f'results/data/{task_name}/{strqtasks}/ESN/{inp_type}').mkdir(parents=True, exist_ok=True)
                    fname = f'results/data/{task_name}/{strqtasks}/ESN/{inp_type}/{pm}_Nesn{N_esn}_sweep_gl_ax_{ax_str}'

                    q_task_dict = {}
                    for i, qtask in enumerate(task.qtasks):
                        q_task_dict['C_mean '+qtask] = C_mean[:,:,i]
                        q_task_dict['C_std '+qtask] = C_std[:,:,i]
                    np.savez_compressed(fname, g_val=g_sweep_val, l_val=l_sweep_val, **q_task_dict)

                else:

                    fname = f'results/data/{task_name}/ESN/{pm}_Nesn{N_esn}_sweep_gl'
                    np.savez_compressed(fname, g_val=g_sweep_val, l_val=l_sweep_val, C_mean=C_mean, C_std=C_std)

            return g_sweep_val, l_sweep_val, C_mean, C_std
        
        else: 

            if g_fixed is None or l_fixed is None:
                raise ValueError("When sweep is False, both g_fixed and l_fixed must be provided.")
            
            C_shape = (len(delays), N_iter, nqtasks) if task_name == 'Qinp' else (len(delays), N_iter)
            C = np.full(C_shape, fill_value=np.nan)

            if not load_data:
                    
                data = EsnDynamics.esn_state_g_l(
                    N_esn, g_sweep_val=[g_fixed], l_sweep_val=[l_fixed], task_name=task_name,
                    qtasks=qtasks, axis=axis, N_iter=N_iter, seed=seed, 
                    rewrite=True, store=False, inp_type=inp_type)
                
            for it in range(N_iter):

                if load_data:

                    data = load_state_data(N_esn, g, l, task_name, it, ax_str=ax_str, inp_type=inp_type)
                    inp = data['inp']
                    states = data['states']
                        
                else:

                    sample = data[it]
                    states = sample[0]
                    inp = sample[1]

                task.x_out = states
                task.input_signals = inp

                if task_name == 'Qinp':
                    task.reset_quantum_input_features()

                for j, n_delay in enumerate(delays):

                    task.n_max_delay = n_delay
                    task.get_output_signal(reshapeflag=True)
                    C[j,it] = task.performance(qflag=True)

                C_mean = np.mean(C, axis=1)
                C_std = np.std(C, axis=1)

            if store:

                if task_name == 'Qinp':

                    strqtasks = task.strqtasks
                    ax_str , _ = ax_to_str(axis=axis, caxis=[])
                    pathlib.Path(f'results/data/{task_name}/{strqtasks}/ESN/{inp_type}').mkdir(parents=True, exist_ok=True)
                    fname = f'results/data/{task_name}/{strqtasks}/ESN/{inp_type}/{pm}_Nesn{N_esn}_g{g_fixed}_l{l_fixed}_ax_{ax_str}_sweep_delay'
                    
                    q_task_dict = {}
                    for i, qtask in enumerate(task.qtasks):
                        q_task_dict['C_mean '+qtask] = C_mean[:,i]
                        q_task_dict['C_std '+qtask] = C_std[:,i]                
                
                    np.savez_compressed(fname, delays=delays, **q_task_dict)
                
                else:

                    fname = f'results/data/{task_name}/ESN/{pm}_Nesn{N_esn}_g{g_fixed}_l{l_fixed}_sweep_delay'
                    np.savez_compressed(fname, delays=delays, C_mean=C_mean, C_std=C_std)

            return delays, C_mean, C_std