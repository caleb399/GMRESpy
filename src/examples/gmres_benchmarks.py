import os
import timeit
import numpy as np
import h5py
from scipy.sparse import csr_matrix
import scipy.sparse.linalg

from GMRESpy.solvers import gmres


def main():
	# Load system
	filename = r'..\..\test_data\step.h5'
	L = read_sparse_matrix_hdf5(filename)
	with h5py.File(filename, 'r') as f:
		b = f['b'][:]

	# GMRES parameters
	n = L.shape[0]
	restart = 80
	max_iter = 1000
	rtol = 1e-3
	x0 = np.zeros(n, dtype = np.float32)

	x0_copy = x0.copy()  # Make sure x0 isn't changed after each call to GMRES

	def print_benchmark(x_, rerr_, start_):
		rerr_check = np.linalg.norm(L @ x_ - b) / np.linalg.norm(b)
		print(f"\tResidual error reported by solver: {rerr_}")
		print(f"\tResidual error calculated directly in gmres.py: {rerr_check}")
		if np.linalg.norm(x0_copy - x0) > 1e-6:
			print('Warning: x0 was changed during call to GMRES')
		end = timeit.default_timer()
		print(f'\tTime: {end - start_} seconds.\n')

	# DPC++ using CPU
	print('\nRunning GMRES benchmark using GMRESpy.solvers.gmres with CPU\n')
	start = timeit.default_timer()
	x, rerr, status = gmres(L, x0, b, n, show_progress = True, rtol = rtol, restart = restart, device = "CPU")
	print_benchmark(x, rerr, start)

	# DPC++ using GPU
	print('Running GMRES benchmark using GMRESpy.solvers.gmres with GPU\n')
	start = timeit.default_timer()
	x, rerr, status = gmres(L, x0, b, n, show_progress = True, rtol = rtol, restart = restart, device = "GPU")
	print_benchmark(x, rerr, start)

	# scipy.sparse.linalg
	print('Running GMRES benchmark using scipy.sparse.linalg\n')
	start = timeit.default_timer()
	x, info = scipy.sparse.linalg.gmres(L, b, x0 = x0, rtol = rtol, restart = restart, maxiter = max_iter)
	print_benchmark(x, None, start)


def read_sparse_matrix_hdf5(filename, dataset_name = 'sparse_matrix'):
	with h5py.File(filename, 'r') as g:
		data = g['data'][:]
		indices = g['indices'][:]
		indptr = g['indptr'][:]
		shape = tuple(g['shape'][:])
	return csr_matrix((data, indices, indptr), shape = shape)


if __name__ == '__main__':
	main()
