## GMRESpy: A DPC++ GMRES Solver

This project provides a Python interface to a DPC++ implementation of Generalized Minimal RESidual (GMRES) algorithm 
for solving sparse linear systems. The implementation is optimized for parallel execution
on GPUs and multicore CPUs.

The Python interface has an optional parameter `target_device` that controls 
whether GMRES runs on CPU or GPU, as follows:
- `target_device = "DEFAULT"`:   Selects the most performant device (chosen by `sycl::default_selector_v`)
- `target_device = "CPU"`:   Selects the CPU
- `target_device = "GPU"`:   Selects the most performant, compatible GPU

## Requirements
The DPC++ documentation [here](https://www.intel.com/content/www/us/en/developer/articles/system-requirements/intel-dpc-compatibility-tool-system-requirements.html) covers most of the system requirements.

Other miscellaneous requirements are listed below.

- Operating system: Windows 10/11
- Python version: 3+ (see `setup.py` for package dependencies)
- [Eigen](https://gitlab.com/libeigen/eigen) and [Pybind11](https://github.com/pybind/pybind11) (included as git submodules).
- To run the example problem in `gmres_examples.dp.cpp`, HDF5 must be installed in the location specified in `CMakelists.txt`.
- Device compatibility: The current CMake build configuration targets Intel CPUs and GPUs.
- Running GMRESpy requires that `sycl7.dll` is in the PATH variable of the environment. This should already be the case for the Intel OneAPI command prompt. Otherwise, `sycl7.dll` can be found in the OneAPI installation directory, e.g. `C:\Program Files (x86)\Intel\oneAPI\2024.0\bin`.
## Installation

In addition to the other requirements, building / installing requires CMake 3.23.0+ and [Ninja](https://ninja-build.org/).

### Direct build (no Python installation)
To build directly without Python, run the following from the Intel OneAPI command
prompt:

```bash
git clone https://github.com/caleb399/GMRESpy.git
cd GMRESpy
mkdir build & cd build
CMake -GNinja ..
ninja all
```
This will build the Python module dependency `gmres_dpcpp.pyd` 
and an executable `dpcpp_gmres.exe` containing an example problem.
### Python setuptools
To install the Python module, it is recommended to use a virtual environment. First open the OneAPI command prompt, and enter:
```bash
python -m venv GMRESpy
.\GMRESpy\Scripts\activate.bat
```
Then, run the following:
```bash
git clone https://github.com/caleb399/GMRESpy.git
cd GMRESPy\GMRESpy
pip install .
cd ..
```
Before using GMRESpy, be sure to delete the original `GMRESpy` directory or navigate to a different directory.
This ensures that Python will correctly load the GMRESpy package from installed site-packages directory.

## Example usage
See `src\examples\gmres_benchmarks.py` for example usage.

## Sample Benchmarks
- Solution of a 524288 x 524288 sparse linear system (test_data\test1.h5)
- Number of nonzeros: 3667968
- GMRES parameters: restart = 80, rtol = 1e-5
- For the scipy benchmarks, OMP_NUM_THREADS was set to either 4 or 8, depending on which one gave the fastest time.

| Library/Environment | Hardware                | Time (seconds) |
|---------------------|-------------------------|----------------|
| GMRESpy             | Integrated Arc Graphics | 151            |
| SciPy + openBLAS    | Intel 10900KF           | 136            |
| SciPy + Intel MKL   | Core Ultra 7 155H       | 99             |
| SciPy + Intel MKL   | Intel 10900KF           | 80             |
| GMRESpy             | Core Ultra 7 155H       | 82             |
| GMRESpy             | Intel 10900KF           | 60             |
| GMRESpy             | Intel Arc 770           | 37             |


## Development Status and To-Do List

Note: This project is in an early stage of development. Features and documentation may be incomplete or subject to change.

To-do list (in no particular order):

- Add pybind11 interface for double precision types (already present in the DPC++ code)
- Add support for additional sparse matrix formats.
- Add build options for targeting NVIDIA and AMD GPUs.
- Improve error handling on the DPC++ side.
