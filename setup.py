import os
import shutil
import glob
import subprocess
import setuptools
from setuptools.command.build_ext import build_ext as SetuptoolsBuildExt
import sysconfig


class NinjaBuild(SetuptoolsBuildExt):
	def run(self):
		try:
			subprocess.check_output(['cmake', '--version'])
		except OSError:
			raise RuntimeError("CMake must be installed to build the following extensions: " +
							   ", ".join(e.name for e in self.extensions))

		self.build_extensions()
		super().run()

	def build_extensions(self):
		for ext in self.extensions:
			self.build_extension(ext)

	def build_extension(self, ext):
		cwd = os.getcwd()
		build_temp = os.path.abspath(self.build_temp)
		cmake_args = ['-GNinja']
		os.makedirs(build_temp, exist_ok = True)
		subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd = build_temp)
		subprocess.check_call(['ninja', 'gmres_pybind'], cwd = build_temp)

		pyd_files = glob.glob(os.path.join(build_temp, 'dpcpp_gmres*.pyd'))
		if len(pyd_files) == 0:
			raise RuntimeError(f'Could not find GMRES .pyd library in build directory ({build_temp})')
		elif len(pyd_files) > 1:
			raise RuntimeError(
				f'Multiple .pyd files were found in build directory ({build_temp}) matching the pattern "dpcpp_gmres*.pyd". Please '
				'remove all unused .pyd files and run setup again')
		else:
			target_lib = pyd_files[0]
		site_packages_dir = sysconfig.get_paths()["purelib"]
		module_install_dir = os.path.join(site_packages_dir, ext.name)
		if not os.path.exists(module_install_dir):
			try:
				os.mkdir(module_install_dir)
			except OSError as e:
				print(f'Could not create installation directory ({module_install_dir}).')
				print(f'Original error message: {e}')
		try:
			shutil.move(target_lib, os.path.join(module_install_dir, "dpcpp_gmres.pyd"))
		except shutil.Error as e:
			print(f'Could not move target library ({target_lib}) to install directory ({module_install_dir}).')
			print(f'Original error message: {e}')


class CMakeExtension(setuptools.Extension):
	def __init__(self, name, sourcedir = ''):
		super().__init__(name, sources = [])
		self.sourcedir = os.path.abspath(sourcedir)


# noinspection PyTypeChecker
setuptools.setup(
	name = 'GMRESpy',
	version = '0.1.0',
	author = 'caleb399',
	description = 'A Python package for solving sparse linear systems on CPU or GPU using the Generalized Minimal Residual Method (GMRES)',
	packages = setuptools.find_packages(),
	ext_modules = [CMakeExtension('GMRESpy', os.path.abspath(os.path.join(os.getcwd(), '.')))],
	cmdclass = {'build_ext': NinjaBuild},
	install_requires = [
		'numpy~=1.26.0',
		'scipy~=1.12.0',
		'h5py~=3.10.0',
		'setuptools~=69.1.1',
		'pydantic~=2.6.3']
)
