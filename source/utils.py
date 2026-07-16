"""Utility functions for quantum state generation, observable indexing, and noise modeling.

This module provides helper functions for state initialization, observable computation,
quantum state reconstruction, and statistical noise injection for quantum reservoir
computing simulations.
"""

import numpy as np
from qutip import sigmax, sigmay, sigmaz, qeye, bell_state, ket2dm, tensor, Qobj, basis, expect, gates, rand_dm

def int_or_float(value):
	"""Convert a value to int or float as appropriate."""
	try:
		return int(value)
	except ValueError:
		return float(value)

def load_observables_data(L, Js, W, h, dt, Vmp, Dmp, N_rep, task_name, it, 
						  inp_type='qubit', back_action=False, monitor_axis='x',
						  meas_strength=None, random_unitary=False):
	"""Load precomputed quantum observable dynamics data.

	Parameters
	----------
	L : int
		Number of qubits.
	Js : float
		Coupling strength scale.
	W : float
		Disorder strength.
	h : float
		Transverse field strength.
	dt : float
		Time evolution step.
	Vmp : int
		Time multiplexing parameter.
	Dmp : int
		Dimension multiplexing parameter.
	N_rep : int
		Number of ESN repetition steps.
	task_name : str
		Task identifier (e.g., 'Qinp', 'NARMA').
	it : int
		Iteration number.
	inp_type : str, optional
		Input type ('qubit', 'werner', 'x_state', etc.).
	back_action : bool, optional
		If True, include weak-measurement back-action data.
	monitor_axis : str, optional
		Measurement axis ('x', 'y', 'z').
	meas_strength : float, optional
		Measurement strength for back-action.

	Returns
	-------
	numpy.lib.npyio.NpzFile
		Observable dynamics loaded from disk.
	"""

	path = f"results/data/"
	if random_unitary:
		path += 'random_unitary/'
	if back_action:
		path += f"back_action/{monitor_axis}/"
		pathend = f"_MeasStr_{meas_strength}"
	
	path += f"{task_name}/QRC/{inp_type}/L{L}_Js{Js}_h{h}_W{W}_dt{dt}_V{Vmp}_D{Dmp}_Nrep{N_rep}"
	if back_action:
		path += pathend
	
	path += f"/Iter_{it}.npz"
	obs = np.load(path, allow_pickle=True)

	return obs

def get_obs_idx(L, axis, caxis, ccaxis, Vmp, Dmp):
	"""Compute indices of observable components for feature extraction.

	Parameters
	----------
	L : int
		Number of qubits.
	axis : list[str]
		Local observable axes to include ('x', 'y', 'z').
	caxis : list[str]
		Two-qubit correlation axes to include.
	ccaxis : list[str]
		Cross-axis correlation axes to include ('zx', 'xy', 'zy').
	Vmp : int
		Time multiplexing parameter.
	Dmp : int
		Dimension multiplexing parameter.

	Returns
	-------
	tuple
		``(idx, q1_idx)`` where ``idx`` is the list of all observable indices
		and ``q1_idx`` is the list of indices for qubit 1 observables.
	"""
	Lc = np.sum(np.arange(L))
	Lc2 = 2*Lc
	idx = []
	q1_idx = [] # indeces of qubit 1

	base = 3 * (L + Lc + Lc2) * Vmp

	for D in range(Dmp):
		offset_D = D * base

		# Local observables
		for ax_index, ax in enumerate(['z', 'x', 'y']):
			if ax in axis:
				for i in range(L):
					for V in range(Vmp):
						val = i + L*V + ax_index * L * Vmp + offset_D
						idx.append(val)
						if i == 0:
							q1_idx.append(val)

		# Correlation observables
		for cax_index, cax in enumerate(['z', 'x', 'y']):
			if cax in caxis:
				for i in range(Lc):
					for V in range(Vmp):
						val = i + Lc*V + (3*L + cax_index*Lc)*Vmp + offset_D
						idx.append(val)
						if i == 0:
							q1_idx.append(val)
		
		# Cross axis correlations observables
		for ccax_index, ccax in enumerate(['zx', 'xy', 'zy']):
			if ccax in ccaxis:
				for i in range(Lc2):
					for V in range(Vmp):
						val = i + Lc2*V + (3*(L+Lc) + ccax_index*Lc2)*Vmp + offset_D
						idx.append(val)
						if i == 0:
							q1_idx.append(val)
						
	return idx, q1_idx

def random_qubit_generator(seed):
	"""Generate a random single-qubit density matrix.

	Parameters
	----------
	seed : int
		Random seed for reproducibility.

	Returns
	-------
	Qobj
		2x2 random single-qubit density matrix.
	"""

	ket0 = basis(2,0)
	ket1 = basis(2,1)

	rng = np.random.default_rng(seed)

	# To generate random eigenvectors we generate a radnom unitary matrix:
	alpha = rng.uniform(0,2*np.pi)
	psi = rng.uniform(0,2*np.pi)
	xi = rng.uniform(0,2*np.pi)
	zeta = rng.uniform(0,1)
	phi = np.arcsin(np.sqrt(zeta))

	u11 = np.cos(phi) * np.exp(complex(0,psi))
	u22 = np.conj(u11)
	u12 = np.sin(phi) * np.exp(complex(0,xi))
	u21 = -np.conj(u12)

	U = np.exp(complex(0,alpha))* Qobj([[u11,u12],[u21,u22]])

	# Generate random eigen vectors
	v0 = U * ket0
	v1 = U * ket1

	# Generate random eigenvalues (probabilities)
	p0 = rng.uniform(0,1)
	p1 = 1 - p0

	rho = p0 * ket2dm(v0) + p1 *ket2dm(v1)
	return rho

def uniform_putiry_state_generator(seed):
	"""Generate a single-qubit state with uniformly distributed purity.

	Parameters
	----------
	seed : int
		Random seed for reproducibility.

	Returns
	-------
	Qobj
		2x2 single-qubit density matrix with uniform purity in [0.5, 1].
	"""

	rng = np.random.default_rng(seed)
	purity = rng.uniform(0.5, 1)
	r = np.sqrt(2*purity-1)

	phi = rng.uniform(0, 2*np.pi)
	theta = rng.uniform(0, np.pi)

	sintheta = np.sin(theta)

	x = r * sintheta * np.cos(phi)
	y = r * sintheta * np.sin(phi)
	z = r * np.cos(theta)

	rho = 0.5 * (qeye(2) + x * sigmax() + y * sigmay() + z * sigmaz())
	return rho

def random_werner_state(p):
	"""Generate a two-qubit Werner state.

	Parameters
	----------
	p : float
		Mixture parameter in [0, 1].
		p=1 yields the Bell state, p=0 yields the maximally mixed state.

	Returns
	-------
	Qobj
		4x4 two-qubit Werner state density matrix.
	"""

	rho = p * ket2dm(bell_state('11')) + (1-p)/4 * qeye([[2,2]])
	return rho

def random_x_state(seed):
	"""Generate a random two-qubit X-state.

	X-states have non-zero elements only on the diagonal and antidiagonal.

	Parameters
	----------
	seed : int
		Random seed for reproducibility.

	Returns
	-------
	Qobj
		4x4 two-qubit X-state density matrix.
	"""

	# Real diagonal elements, sampled randomly and normalized (trace 1)
	rng = np.random.default_rng(seed=seed)

	diag = rng.random(4)
	diag /= np.sum(diag)
	a, b, c, d = diag 

	assert (np.sum(diag) - 1 < 1e-10)

	# Random complex off-diagonal terms
	max_w = np.sqrt(a * d)  # constraint from positivity
	max_z = np.sqrt(b * c)

	min_w = 0 ; min_z = 0

	abs_w = rng.uniform(min_w, max_w)
	abs_z = rng.uniform(min_z, max_z)
	phase_w = rng.uniform(0, 2 * np.pi)
	phase_z = rng.uniform(0, 2 * np.pi)

	w = abs_w * np.exp(1j * phase_w)
	z = abs_z * np.exp(1j * phase_z)

	# Construct the X-state matrix
	rho = np.array([
		[a, 0,   0,   w],
		[0, b,   z,   0],
		[0, z.conjugate(), c, 0],
		[w.conjugate(), 0, 0, d]
	], dtype=complex)

	rho = Qobj(rho, dims=[[2,2],[2,2]])

	return rho

def load_readout_layer(L, Js, W, h, dt, Vmp, Dmp, N_rep, N_esn, g, l, task_name, it):

	""" Load the qrc observable and esn states dynamics computed with the satticmethod parallel_hybrid_worker"""

	path = f'results/data/{task_name}/HYB/L{L}_Js{Js}_h{h}_W{W}_dt{dt}_Vmp{Vmp}_Dmp{Dmp}_Nrep{N_rep}_Nesn{N_esn}_g{g}_l{l}/Iter_{it}.npz'
	state = np.load(path)

	return state

def ax_to_str(axis, caxis):
	"""Convert axis lists to concatenated string representation.

	Parameters
	----------
	axis : list[str]
		Local observable axes.
	caxis : list[str]
		Correlation observable axes.

	Returns
	-------
	tuple
		``(ax_str, cax_str)`` where each is a concatenated string of axes.
	"""
	ax_str = ''
	cax_str = ''
	
	for i in axis:
		ax_str += i
	for j in caxis:
		cax_str += j

	return ax_str, cax_str

def reconstruct_rho(sz, sx, sy):
	"""Reconstruct single-qubit density matrix from Pauli expectation values.

	Parameters
	----------
	sz : float
		Expectation value of Z Pauli operator.
	sx : float
		Expectation value of X Pauli operator.
	sy : float
		Expectation value of Y Pauli operator.

	Returns
	-------
	Qobj
		2x2 reconstructed single-qubit density matrix.

	Notes
	-----
	Assumes the standard Bloch sphere parametrization: :math:`\\rho = \\frac{1}{2}(I + s_x\\sigma_x + s_y\\sigma_y + s_z\\sigma_z)`.
	"""

	rho = 0.5 * (qeye(2) + sz * sigmaz() + sx * sigmax() + sy *sigmay())
	
	return rho

def reconstruct_2qubit(row):
	"""Reconstruct two-qubit density matrix from Pauli expectation values.

	Parameters
	----------
	row : array-like
		1D array with 15 expectation values [xI, Ix, yI, Iy, zI, Iz, xx, xy, xz, yx, yy, yz, zx, zy, zz]
		corresponding to two-qubit Pauli operator expectations.

	Returns
	-------
	Qobj
		4x4 reconstructed two-qubit density matrix.

	Notes
	-----
	Reconstructs the density matrix from expectation values of all two-qubit Pauli products
	except the identity ⊗ identity term.
	"""
	# Define single-qubit Pauli operators
	paulis = [qeye(2), sigmax(), sigmay(), sigmaz()]
	ops = []

	for i, A in enumerate(paulis):
		for j, B in enumerate(paulis):
			if not (i == 0 and j == 0):  # skip I\otimesI term
				ops.append(tensor(A, B))

	# Start with identity contribution
	rho = tensor(qeye(2), qeye(2))

	# Add weighted Pauli terms
	for val, op in zip(row, ops):
		rho += val * op

	rho = rho / 4
	return rho

def input_full_esn(inp, inp_type, axis=None):
	"""Extract Pauli observable expectation values from quantum states.

	Parameters
	----------
	inp : list[Qobj] or array-like
		Quantum states (Qobj) or classical features (array).
	inp_type : str
		Input type: 'qubit' (single-qubit), 'werner', 'x_state', '2qubit', 'rand_bell_mix', etc.
	axis : list[str], optional
		Observable axes to extract ('x', 'y', 'z').
		Defaults to empty list.

	Returns
	-------
	ndarray
		Array of shape (n_states, n_observables) containing expectation values.

	Raises
	------
	ValueError
		If invalid axes are provided for multi-qubit states without valid axis selection.
	"""

	if axis is None:
		axis = []

	axis_op = {
		'z': sigmaz(),
		'x': sigmax(),
		'y': sigmay()
	}

	n = len(inp)

	if inp_type == 'qubit':
		a = np.zeros((n, len(axis)))
		for k, ax in enumerate(axis):
			a[:, k] = np.array([expect(axis_op[ax], i) for i in inp])

	elif inp_type in ['werner', 'x_state', '2qubit', 'rand_bell_mix', '2qubit_pure', '2qubit_rank2']:

		# validate axes (silently ignore unknown)
		axis = [ax for ax in axis if ax in axis_op]
		if len(axis) == 0:
			raise ValueError("No valid axes provided (choose from 'x','y','z').")

		ops = []
		labels = []

		# local operators: sigma_a ⊗ I  and I ⊗ sigma_a
		for ax in axis:
			ops.append(tensor(axis_op[ax], qeye(2)))
			labels.append(f"{ax}I")
			ops.append(tensor(qeye(2), axis_op[ax]))
			labels.append(f"I{ax}")

		# correlators: sigma_a ⊗ sigma_b for all pairs (a,b) in axes x axes
		for ax1 in axis:
			for ax2 in axis:
				ops.append(tensor(axis_op[ax1], axis_op[ax2]))
				labels.append(f"{ax1}{ax2}")

		n_ops = len(ops)

		a = np.full((n, n_ops), np.nan)

		# Use qutip.expect with list of states for each operator (fast)
		for i, op in enumerate(ops):
			a[:, i] = np.real(expect(op, list(inp)))

	else:
		raise ValueError(f"Unknown inp_type: {inp_type}")

	return a

def load_state_data(N_esn, g, l, task_name, it, ax_str='', inp_type='qubit'):

	""" Load the state dynamics data computed with the static method esn_worker """

	if task_name == 'Qinp':
		path = f'results/data/{task_name}/ESN/ESN_Nesn{N_esn}_g{g}_l{l}_ax_{ax_str}_{inp_type}/Iter_{it}.npz'
	else:
		path = f'results/data/{task_name}/ESN/ESN_Nesn{N_esn}_g{g}_l{l}/Iter_{it}.npz'

	state = np.load(path)

	return state

def ensure_physical(rho, tol=1e-12):
	"""Check if a density matrix is physical (positive semidefinite with trace 1).

	Parameters
	----------
	rho : Qobj
		Quantum density matrix to validate.
	tol : float, optional
		Tolerance for positivity and trace checks.

	Returns
	-------
	Qobj or None
		Returns ``rho`` if physical, otherwise ``None``.
	"""
	evals = rho.eigenenergies()  # cheaper than eigenstates
	if (np.min(evals) >= -tol) and (abs(rho.tr() - 1.0) < tol):
		return rho
	else:
		return None

def get_M_Had_HS_operators(monitor_axis, meas_strength, L):
	"""Construct back-action operators for weak measurement monitoring.

	Parameters
	----------
	monitor_axis : str
		Measurement axis ('x', 'y', 'z').
	meas_strength : float
		Strength of the weak measurement.
	L : int
		Number of qubits.

	Returns
	-------
	tuple
		``(M, Had, Ry)`` where M is the measurement back-action operator,
		Had is the Hadamard gate (None for z-axis), and Ry is the Y-axis rotation
		(None for z/x axes).
	"""

	sup = np.exp(-meas_strength**2/2)
	M_qubit = np.array([[1, sup], [sup, 1]])
	QM_qubit = Qobj(M_qubit)
	QM = tensor(QM_qubit for _ in range(L))
	M = QM.full()

	if monitor_axis in ['x', 'y']:
		Had_qubit = gates.snot()
		Had = tensor(Had_qubit for _ in range(L))
	else:
		Had = None

	if monitor_axis == 'y':
		S_qubit = gates.phasegate(np.pi / 2)
		S = tensor(S_qubit for _ in range(L))
		Ry = Had * S.dag()
	else:
		Ry = None

	return M, Had, Ry


def monitor_rho_transform(rho, monitor_axis, M, Had=None, Ry=None):
	"""Apply weak-measurement back-action transformation to density matrix.

	Parameters
	----------
	rho : Qobj
		Quantum state after CPTP map (unitary evolution or free-noise map).
	monitor_axis : str
		Measurement axis ('x', 'y', or 'z').
	M : array
		Back-action measurement operator.
	Had : Qobj, optional
		Hadamard gate for x-axis rotation.
	Ry : Qobj, optional
		Combined Hadamard and phase shift for y-axis rotation.

	Returns
	-------
	Qobj
		Density matrix after continuous weak-measurement back-action.

	Raises
	------
	ValueError
		If monitor_axis is not 'x', 'y', or 'z'.

	Notes
	-----
	The transformation depends on the measurement axis:
	- 'z': Direct application of measurement operator
	- 'x': Rotation to z-basis, application, rotation back
	- 'y': Y-basis rotation, application, rotation back
	"""

	if monitor_axis not in ['z', 'x', 'y']:
		raise ValueError("Monitor axis must be one of the following strings: 'x', 'y', 'z'.")
	elif monitor_axis == 'z':
		rho_ba = Qobj(np.multiply(M,rho.full()), dims=rho.dims)
	elif monitor_axis == 'x':
		rho_rotx = Had * rho * Had
		rho_ba = Had * Qobj(np.multiply(M, rho_rotx.full()), dims=rho.dims) * Had
	elif monitor_axis == 'y':
		rho_roty = Ry * rho * Ry.dag()
		rho_ba = Ry.dag() * Qobj(np.multiply(M, rho_roty.full()), dims=rho.dims) * Ry

	return rho_ba


import math
def orderOfMagnitude(number):
	return math.floor(math.log(number, 10))


def statistical_noise(obs, N_meas, meas_strength, L, V, back_action, seed=34):
	"""Add statistical noise to observable measurements.

	Parameters
	----------
	obs : ndarray
		Ideal (noiseless) observables of shape (n_timesteps, n_observables).
	N_meas : int
		Number of measurement repetitions (shots).
	meas_strength : float
		Measurement strength parameter (nonzero required).
	L : int
		Number of qubits.
	V : int
		Time multiplexing parameter.
	back_action : bool
		If True, apply noise model accounting for back-action; otherwise apply uniform noise.
	seed : int, optional
		Random seed for reproducibility.

	Returns
	-------
	tuple
		``(noisy_obs, noise_order_of_magnitude)`` where ``noisy_obs`` is the
		noise-corrupted array and ``noise_order_of_magnitude`` is the exponent
		of 10 for local observable noise level.

	Raises
	------
	ValueError
		If ``meas_strength`` is zero (noise would diverge).
	"""

	T = obs.shape[0]
	noisy_obs = obs.copy()
	LV = L*V
	rng = np.random.default_rng(seed)

	if not meas_strength:
		raise ValueError("Error diverges to infinite, choose a nonzero g value")

	if back_action:

		g4 = meas_strength ** 4
		g2 = meas_strength ** 2

		one_obs_noise = np.sqrt((g2 + 1)/(g2 * N_meas))
		two_corr_noise = np.sqrt((g4 + 2*g2 + 1)/(g4 * N_meas))

		noisy_obs[:, :LV] += rng.normal(0, one_obs_noise, size=(T, LV))
		noisy_obs[:, LV:] += rng.normal(0, two_corr_noise, size=(T, obs.shape[1]-LV))

	else:

		all_obs_noise = 1/np.sqrt(N_meas)
		noisy_obs += rng.normal(0, all_obs_noise, size=obs.shape)

	return noisy_obs, orderOfMagnitude(one_obs_noise)

def concurrence_not_max(rho):
	"""Compute Wootters concurrence for a two-qubit state.

	Parameters
	----------
	rho : Qobj
		4x4 two-qubit density matrix.

	Returns
	-------
	float
		Non maximized Concurrence value.
	"""
	sy = sigmay()
	Y = tensor(sy, sy)

	# spin-flipped state
	rho_tilde = Y * rho.conj() * Y

	# R = sqrt( sqrt(rho) * rho_tilde * sqrt(rho) )
	sqrt_rho = rho.sqrtm()
	R = (sqrt_rho * rho_tilde * sqrt_rho).sqrtm()

	# eigenvalues of R in decreasing order
	evals = np.sort(np.real(R.eigenenergies()))[::-1]

	# Wootters concurrence
	C = evals[0] - evals[1] - evals[2] - evals[3]
	return C

def rand_bell_mixture(p, seed):
	"""Generate a mixture of Bell state and random two-qubit state.

	Parameters
	----------
	p : float
		Mixture parameter in [0, 1].
		p=1 yields the Bell state, p=0 yields random state.
	seed : int
		Random seed for the random state component.

	Returns
	-------
	Qobj
		4x4 two-qubit mixed density matrix.
	"""

	bell = bell_state('11')
	Bell = bell * bell.dag()
	rho = p * rand_dm(2*[2],seed=seed) + (1-p) * Bell
	return rho

def Nmeas_RegPrepParam(V, noise_ofm):
	"""Compute regularization parameter based on noise and multiplexing.

	Parameters
	----------
	V : int
		Time multiplexing parameter.
	noise_ofm : float
		Order of magnitude (power of 10) of the noise level.

	Returns
	-------
	float
		Computed regularization parameter.
	"""

	var = 10**(noise_ofm); var2 = var * var
	# value = 1e6 * var2 * V
	value = 2000 * var2
	return value
