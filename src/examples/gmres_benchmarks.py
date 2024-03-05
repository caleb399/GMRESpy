import os
import timeit
import numpy as np
import h5py
from scipy.sparse import csr_matrix
import scipy.sparse.linalg

os.add_dll_directory(r'C:\Program Files (x86)\Intel\oneAPI\compiler\2024.0\bin')

from .lib.dpcpp_gmres import dpcpp_gmres

def main():
	# Load system
	L = read_sparse_matrix_hdf5('step.h5')
	with h5py.File('step.h5', 'r') as f:
		b = f['b'][:]

	# GMRES parameters
	n = L.shape[0]
	restart = 80
	max_iter = 1000
	rtol = 1e-3
	x0 = np.zeros(n, dtype = np.float32)
	rerr = -1

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
	print('Running GMRES benchmark using dpcpp_gmres.gmres with CPU')
	start = timeit.default_timer()
	x, rerr, status = dpcpp_gmres.gmres(L.data.astype(np.float32), L.indices.astype(np.int32),
										L.indptr.astype(np.int32),
										x0, b, rerr, n, show_progress = True, rtol = rtol, restart = restart, device = "CPU")
	print_benchmark(x, rerr, start)

	# DPC++ using GPU
	print('Running GMRES benchmark using dpcpp_gmres.gmres with GPU')
	start = timeit.default_timer()
	x, rerr, status = dpcpp_gmres.gmres(L.data.astype(np.float32), L.indices.astype(np.int32),
										L.indptr.astype(np.int32),
										x0, b, rerr, n, show_progress = True, rtol = rtol, restart = restart, device = "GPU")
	print_benchmark(x, rerr, start)

	# scipy.sparse.linalg using openMP (if available)
	print('Running GMRES benchmark using scipy.sparse.linalg')
	start = timeit.default_timer()
	x, info= scipy.sparse.linalg.gmres(L, b, x0 = x0, rtol = rtol, restart = restart, maxiter = max_iter)
	print_benchmark(x, -1, start)


def read_sparse_matrix_hdf5(filename, dataset_name = 'sparse_matrix'):
	with h5py.File(filename, 'r') as g:
		data = g['data'][:]
		indices = g['indices'][:]
		indptr = g['indptr'][:]
		shape = tuple(g['shape'][:])
	return csr_matrix((data, indices, indptr), shape = shape)


if __name__ == '__main__':
	main()
