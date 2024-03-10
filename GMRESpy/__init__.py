import sys
import os

sycl_dll_found = False
sycl7_dll_path = ''

# Give priority to dll in INTEL_SYCL_DIR, if it exists
if 'INTEL_SYCL_DIR' in os.environ:
	test_path = os.path.join(os.environ['INTEL_SYCL_DIR'], 'sycl7.dll')
	if os.path.isfile(test_path):
		sycl_dll_found = True
		sycl7_dll_path = os.environ['INTEL_SYCL_DIR']

# Otherwise, check 'CMPLR_ROOT' environment variable.
# 'CMPLR_ROOT' should be set by 'setvars.bat' in the Intel OneAPI install directory.
if not sycl_dll_found:
	if 'CMPLR_ROOT' in os.environ:
		test_path = os.path.join(os.environ['CMPLR_ROOT'], 'bin', 'sycl7.dll')
		if os.path.isfile(test_path):
			sycl_dll_found = True
			sycl7_dll_path = os.path.join(os.environ['CMPLR_ROOT'], 'bin')

if sycl_dll_found and os.path.exists(sycl7_dll_path):
	try:
		os.add_dll_directory(sycl7_dll_path)
	except OSError as e:
		print(f'Error adding DLL directory: {e}')
		sys.exit(-1)

try:
	from .dpcpp_gmres import gmres
except ImportError:
	print('Could not load gmres from shared library. Possible causes include:\n'
		  '1. You are running Python from the same directory where you originally installed GMRESpy. '
			'If this is the case, try switching to another directory.\n'
		  '2. Not all dependent DLLs are on the system path. Loading the Intel OneAPI environment by executing'
		  '"$(ONEAPI_INSTALL_DIR)\setvars.bat" may fix this issue. You can also manually set environment '
		  'variable "INTEL_SYCL_DIR" to the location of sycl7.dll, which is a required dependency.\n'
		  '3. You are using a version of Python that is different from the one used to build GMRESpy.')
	sys.exit(-1)

