/**
 * @file gmres_bindings.dp.cpp
 * @brief Python interface to the GMRES algorithm using 
 * Data Parallel C++ (DPC++ or DPCPP).
 */

#include <numpy.h>
#include <pybind11.h>
#include "gmres.dp.hpp"

namespace py = pybind11;

/**
 * Python interface to a DPC++ implementation of GMRES for solving
 * the linear equation Ax = b, where A is a sparse matrix.
 *
 * Parameters:
 * - values: A py::array_t<float> representing the non-zero values of the sparse matrix A.
 * - col_indices: A py::array_t<int32_t> containing the column indices corresponding to the values in 'values'.
 * - row_offsets: A py::array_t<int32_t> indicating the start of each row in the 'values' array.
 * - x0: Initial guess for the solution vector x, as a py::array_t<float>.
 * - b: The right-hand side vector of the equation Ax = b, as a py::array_t<float>.
 * - n: Size of the solution vector (and the dimension of the square matrix A).
 * - rtol: Relative tolerance for convergence.
 * - restart: Number of iterations after which the algorithm restarts with the current solution as the new initial guess.
 * - maxiter: Maximum number of restarts to perform before stopping.
 * - device: String specifying the target device for DPC++ execution (e.g., "GPU", "CPU"). "DEFAULT" uses the SYCL default device.
 * - show_progress: Boolean indicating whether to print device information to console
 * - show_progress: Boolean indicating whether to print progress messages to console.
 *
 * Returns:
 * A tuple containing the solution vector x as a py::array_t<float>, the final relative error, and the exit code.
 */
PYBIND11_MODULE(dpcpp_gmres, m) {
    m.def(
        "gmres",
        [](py::array_t<float> values,
           py::array_t<int32_t> col_indices,
           py::array_t<int32_t> row_offsets,
           py::array_t<float> x0,
           py::array_t<float> b,
           int32_t n,
           float rtol = 1e-6,
           int32_t restart = 40,
           int32_t maxiter = 1000,
           py::str device = "DEFAULT",
           bool show_platform_info = true,
           bool show_progress = false) {
            // Create copy of x0 so the data doesn't get overwritten
            std::vector<float> x0_copy(n, 0.0f);
            float *x0_data = x0.mutable_data();
            std::copy(x0_data, x0_data + n, x0_copy.begin());

            // Allocate memory for the solution vector
            py::array_t<float> x_solution = py::array_t<float>(n);
            float *x_solution_data = x_solution.mutable_data();

            // Create SYCL buffers
            sycl::buffer<float, 1> values_buf(values.mutable_data(), sycl::range<1>(values.size()));
            sycl::buffer<int, 1> col_indices_buf(col_indices.mutable_data(), sycl::range<1>(col_indices.size()));
            sycl::buffer<int, 1> row_offsets_buf(row_offsets.mutable_data(), sycl::range<1>(row_offsets.size()));
            sycl::buffer<float, 1> x0_buf(x0_copy.data(), sycl::range<1>(n));
            sycl::buffer<float, 1> b_buf(b.mutable_data(), sycl::range<1>(b.size()));

            CSRMatrix<float> A(values_buf, col_indices_buf, row_offsets_buf);

            GmresParams params;
            params.rtol = rtol;
            params.max_iter = maxiter;
            params.restart = restart;
            params.target_device = device;
            params.show_platform_info = show_platform_info;
            params.show_progress = show_progress;

            float rerr;
            auto status = gmres<float>(A,
                                       x0_buf,
                                       b_buf,
                                       x_solution_data,
                                       rerr,
                                       n,
                                       params);

            if (show_progress) {
                if (status > 0)
                    std::cout << "GMRES converged after " << status << " iterations with relative error of " << rerr << "." << std::endl;
                else if (status == -1) {
                    std::cout << "GMRES reached the maximum number of iterations (relative error: " << rerr << ")." << std::endl;
                } else {
                    std::cout << "GMRES exited without performing any iterations (relative error: " << rerr << ")." << std::endl;
                }
            }

            return py::make_tuple(x_solution, rerr, static_cast<int32_t>(status));
        },

        py::arg("values"),
        py::arg("col_indices"),
        py::arg("row_offsets"),
        py::arg("x0"),
        py::arg("b"),
        py::arg("n"),
        py::arg("rtol") = 1e-6,
        py::arg("restart") = 40,
        py::arg("maxiter") = 1000,
        py::arg("device") = "DEFAULT",
        py::arg("show_platform_info") = true,
        py::arg("show_progress") = false);
}
