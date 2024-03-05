import os
import sys

if 'INTEL_SYCL_DIR' in os.environ:
	os.add_dll_directory(os.environ['INTEL_SYCL_DIR'])

try:
	from .lib import gmres
except ImportError:
	if 'INTEL_SYCL_DIR' not in os.environ:
		print('Could not load dpcpp_gmres from shared library. Please set environment variable "INTEL_SYCL_DIR" '
			  'to the location of "sycl7.dll".')
		sys.exit(-1)

try:
	from .lib import gmres
except ImportError as e:
	print(f'Could not load dpcpp_gmres from the shared library dpcpp_gmres.pyd. Ensure you are running with the same '
		  'version of Python used when installing the library. Also, check that all dependent DLLs are on the system path.'
		  'Original error message: {e}')
	sys.exit(-1)
