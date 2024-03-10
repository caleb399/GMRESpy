from pydantic import Field, ValidationError, field_validator
from pydantic import BaseModel as PydanticBaseModel
import numpy as np
from scipy.sparse import csr_matrix

from . import gmres_pybind

from pydantic import BaseModel as PydanticBaseModel

# Required for using scipy.sparse.csr_matrix in pydantic model
class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True

# Pydantic class for validating inputs to GMRES
class SparseSolverParams(BaseModel):
	A: csr_matrix = Field(...)
	x0: np.ndarray = Field(...)
	b: np.ndarray = Field(...)
	n: int = Field(gt = 0)
	rtol: float = Field(default = 1e-6, gt = 0, lt = 1)
	restart: int = Field(gt = 0)
	max_iter: int = Field(gt = 0)
	device: str = Field(default = "DEFAULT")
	show_platform_info: bool = Field(default = True)
	show_progress: bool = Field(default = False)

	@field_validator('A')
	def check_csr_matrix(cls, v):
		if not isinstance(v, csr_matrix):
			raise ValueError('A must be a scipy.sparse.csr_matrix')
		if v.data.dtype != np.float32:
			raise ValueError('A.data must be of dtype float32')
		if v.indices.dtype != np.int32 or v.indptr.dtype != np.int32:
			raise ValueError('A.indices and A.indptr must be of dtype int32')
		return v

	@field_validator('x0', 'b')
	def check_float32(cls, v):
		if v.dtype != np.float32:
			raise ValueError('must be a float32 array')
		return v

	@field_validator('device')
	def check_device(cls, v):
		if v not in ['DEFAULT', 'CPU', 'GPU']:
			raise ValueError('must be DEFAULT, CPU, or GPU')
		return v


def gmres(A, x0, b, n, rtol = 1e-6, restart = 40, max_iter = 1000, device = "DEFAULT", show_progress = False, show_platform_info = True):
	# Validate input parameters
	try:
		validated_params = SparseSolverParams(
			A = A,
			x0 = x0,
			b = b,
			n = n,
			rtol = rtol,
			restart = restart,
			max_iter = max_iter,
			device = device,
			show_platform_info = show_platform_info,
			show_progress = show_progress
		)
	except ValidationError as e:
		print(e)
		print(f'\nFunction arguments did not pass validation; see the above error message for details. (File: {__file__})')
		return None, None, None
	else:
		# params passed check, so now call the DPC++ gmres function (gmres_pybind)
		
		# extract csr matrix fields to pass directly to gmres_pybind
		values = validated_params.A.data.astype(np.float32)
		indices = validated_params.A.indices.astype(np.int32)
		indptr = validated_params.A.indptr.astype(np.int32)

		solution, final_error, status_code = gmres_pybind(
			values,
			indices,
			indptr,
			validated_params.x0,
			validated_params.b,
			np.int32(validated_params.n),
			rtol = np.float32(validated_params.rtol),
			restart = np.int32(validated_params.restart),
			maxiter = np.int32(validated_params.max_iter),
			device = validated_params.device,
			show_platform_info = validated_params.show_platform_info,
			show_progress = validated_params.show_progress)

	return solution, final_error, status_code


if __name__ == "__main__":
	print(f'Modules imported successfully (file: {__file__})')
