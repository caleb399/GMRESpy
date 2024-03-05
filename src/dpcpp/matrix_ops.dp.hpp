/**
 * @file matrix_ops.dp.hpp
 * @brief Basic matrix operations for use with GMRES. Except for `solve_upper_hessenberg`, all functions support
 * parallel execution using SYCL.
*/

#ifndef DPCPP_GMRES__MATRIX_OPS_DP_HPP
#define DPCPP_GMRES__MATRIX_OPS_DP_HPP

#include <CL/sycl.hpp>
#include <Eigen/Dense> // For Hessenberg solver
#include "csr.dp.hpp" // For `CSRMatrix<T>` struct definition

/**
 * Solves a linear system Ax = b where A is an upper Hessenberg matrix.
 *
 * Uses the QR decomposition method from "Eigen" to solve
 * the linear system. The matrix A (in this case, an upper Hessenberg matrix)
 * and vector b should be passed in as flattened arrays (std::vector<double>).
 *
 * @param H_data The flattened array representing the upper Hessenberg matrix A. The matrix
 *               is expected to be stored in row-major order and its size must be (m+1)*m.
 * @param b_data The vector b as a flattened array, with its size expected to be m+1.
 * @param m The number of columns in the upper Hessenberg matrix A.
 *
 * @return A std::vector<double> containing the solution vector x to the linear system Ax = b.
 */
template<typename T>
std::vector<T> solve_upper_hessenberg(const std::vector<T> &H_data,
                                         const std::vector<T> &b_data,
                                         int m) {
    if (H_data.size() != (m + 1) * m || b_data.size() != m + 1) {
        throw std::invalid_argument("Invalid sizes for H_data or b_data.");
    }

    Eigen::MatrixXd H = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(H_data.data(), m + 1, m);
    Eigen::VectorXd b = Eigen::Map<const Eigen::VectorXd>(b_data.data(), m + 1);
    Eigen::VectorXd x = H.householderQr().solve(b);
    std::vector<T> x_vec(x.data(), x.data() + x.size());
    return x_vec;
}

/**
 * @brief Multiplies an n x n CSR matrix `A` by the vector `x`, storing the result in vector `y`.
 *
 * @tparam T The data type of the elements in the matrix and the vector.
 * @param q The SYCL queue.
 * @param A The CSRMatrix object representing the sparse matrix. Must have member buffers: values_buf, col_indices_buf, and row_offsets_buf.
 * @param x_buf The input buffer containing the vector `x` to be multiplied.
 * @param y_buf The output buffer where the result vector `y` will be stored.
 * @param n The number of rows in matrix `A` (and also the size of vectors `x` and `y`).
 */
template <typename T>
void csr_matrix_vector_mult(sycl::queue &q, CSRMatrix<T>& A,
                            sycl::buffer<T, 1> &x_buf,
                            sycl::buffer<T, 1> &y_buf,
                            size_t n) {
    q.submit([&](sycl::handler &h) {
         auto values = A.values_buf.template get_access<sycl::access::mode::read>(h);
         auto col_indices = A.col_indices_buf.template get_access<sycl::access::mode::read>(h);
         auto row_offsets = A.row_offsets_buf.template get_access<sycl::access::mode::read>(h);
         auto x = x_buf.template get_access<sycl::access::mode::read>(h);
         sycl::accessor y(y_buf, h, sycl::write_only, sycl::no_init);
         h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
             size_t row = idx[0];
             T dot_product = static_cast<T>(0);
             for (size_t j = row_offsets[row]; j < row_offsets[row + 1]; ++j) {
                 dot_product += values[j] * x[col_indices[j]];
             }
             y[row] = dot_product;
         });
     }).wait();
}

/**
 * @brief Perform matrix-vector multiplication using a CSR sparse matrix,
 * where the vector is a row from a dense matrix `X` and the result is stored 
 * in the next row of `X`. This is an overload in which `X` is given as a sycl::device_vector.
 *
 * @tparam T The data type of the elements in the matrix and the vector.
 * @param q SYCL queue.
 * @param matrix The CSR matrix to be multiplied.
 * @param X Pointer (device vector) to the matrix X.
 * @param n Matrix dimensions and row dimension of X.
 * @param row_idx The row index in X corresponding to the vector to be multiplied. The result vector is stored in the next row of X.
 */
template <typename T>
void csr_matrix_vector_mult(sycl::queue &q, CSRMatrix<T>& A,
                            T* X,
                            size_t n,
                            size_t row_idx) {
    q.submit([&](sycl::handler &h) {
        auto values = A.values_buf.template get_access<sycl::access::mode::read>(h);
        auto col_indices = A.col_indices_buf.template get_access<sycl::access::mode::read>(h);
        auto row_offsets = A.row_offsets_buf.template get_access<sycl::access::mode::read>(h);
         h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
             size_t row = idx[0];
             T dot_product = static_cast<T>(0);
             for (size_t j = row_offsets[row]; j < row_offsets[row + 1]; ++j) {
                 dot_product += values[j] * X[n * row_idx + col_indices[j]];
             }
             X[n * (row_idx + 1) + row] = dot_product;
         });
     }).wait();
}

/**
 * @brief Perform matrix-vector multiplication using a CSR sparse matrix,
 * where the vector is a row from a dense matrix `X` and the result is stored 
 * in the next row of `X`. This is an overload in which `X` is given as a sycl::buffer.
 *
 * @tparam T The data type of the elements in the matrix and the vector.
 * @param q SYCL queue.
 * @param matrix The CSRMatrix object representing the sparse matrix. It must have member buffers: values_buf, col_indices_buf, and row_offsets_buf.
 * @param X_buf Buffer containing the dense matrix `X`.
 * @param n The number of rows in matrix `A` (and also the number of columns in matrix `X` since a row of `X` is used as a vector).
 * @param row_idx The index of the row in `X` to be used as the vector for multiplication. The result is stored in the next row (`row_idx + 1`).
 */
template <typename T>
void csr_matrix_vector_mult(sycl::queue &q,
                            CSRMatrix<T>& A,
                            sycl::buffer<T, 1>& X_buf,
                            size_t n,
                            size_t row_idx) {
    q.submit([&](sycl::handler &h) {
        auto values = A.values_buf.template get_access<sycl::access::mode::read>(h);
        auto col_indices = A.col_indices_buf.template get_access<sycl::access::mode::read>(h);
        auto row_offsets = A.row_offsets_buf.template get_access<sycl::access::mode::read>(h);
        auto X = X_buf.template get_access<sycl::access::mode::read_write>(h);
         h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
             size_t row = idx[0];
             T dot_product = 0.0;
             for (size_t j = row_offsets[row]; j < row_offsets[row + 1]; ++j) {
                 dot_product += values[j] * X[n * row_idx + col_indices[j]];
             }
             X[n * (row_idx + 1) + row] = dot_product;
         });
     }).wait();
}

// Same as `csr_matrix_vector_mult` but using groups and local memory. The matrix `A` must have less than 8 nonzero 
// elements per row to faciliate loading into local memory.
template <typename T>
void csr_matrix_vector_mult_local(sycl::queue &q,
                                  CSRMatrix<T>& matrix,
                                  T* X,
                                  size_t n,
                                  size_t row_idx) {
    const int rows_per_group = q.get_device().get_info<sycl::info::device::max_work_group_size>();
    const int num_groups = (n + rows_per_group - 1) / rows_per_group;
    const int global_size = num_groups * rows_per_group;
    const int X_lm_size = 8; // Assuming <= 8 nonzero matrix elements per row
    if (do_print) {
        std::cout << "Local Size: " << rows_per_group << std::endl;
        std::cout << "Rows per Group: " << rows_per_group << std::endl;
        std::cout << "Number of Groups: " << num_groups << std::endl;
        std::cout << "Global Size: " << global_size << std::endl;
        do_print = false;
    }

    q.submit([&](sycl::handler &h) {
        auto values = A.values_buf.template get_access<sycl::access::mode::read>(h);
        auto col_indices = A.col_indices_buf.template get_access<sycl::access::mode::read>(h);
        auto row_offsets = A.row_offsets_buf.template get_access<sycl::access::mode::read>(h);
        sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local> local_X(sycl::range<1>(rows_per_group * X_lm_size), h);
        sycl::accessor<int32_t, 1, sycl::access::mode::read_write, sycl::access::target::local> local_row_offsets(sycl::range<1>(rows_per_group + 1), h);

        h.parallel_for<class csr_kernel_updated>(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(rows_per_group)), [=](sycl::nd_item<1> item) {
            int global_id = item.get_global_linear_id();
            int local_id = item.get_local_linear_id();
            int group_id = item.get_group_linear_id();

            // Calculate the starting and ending row for this workgroup
            int row_start = group_id * rows_per_group;
            int row_end = sycl::min(row_start + rows_per_group, static_cast<int>(n));

            // Load the relevant portion of row_offsets into local memory
            if (local_id < rows_per_group - 1) {
                local_row_offsets[local_id] = row_offsets[row_start + local_id];
            }
            else if (local_id == rows_per_group - 1) {
                local_row_offsets[rows_per_group - 1] = row_offsets[row_start + local_id];
                local_row_offsets[rows_per_group] = row_offsets[row_start + local_id + 1];
            }
            item.barrier(sycl::access::fence_space::local_space);
            
            int counter = 0;
            for (int j = local_row_offsets[local_id]; j < local_row_offsets[local_id + 1]; ++j) {
                local_X[X_lm_size * local_id + counter] = X[n * row_idx + col_indices[j]];
                counter++;
            }
            item.barrier(sycl::access::fence_space::local_space);
            if (global_id < n) {
                int row = row_start + local_id;
                T dot_product = static_cast<T>(0);;
                int counter = 0;
                for (int j = local_row_offsets[row - row_start]; j < local_row_offsets[row - row_start + 1]; ++j) {
                    dot_product += values[j] * local_X[X_lm_size * local_id + counter];
                    counter++;
                }
                X[n * (row_idx + 1) + row] = dot_product;
            }
        });
    }).wait();
}

/**
 * Perform dense matrix-vector multiplication where the input matrix, input vector, and output vector are represented
 * as 1D sycl::buffers.
 * Note: if the matrix has more rows than elements in the vector, then the
 * extra rows are ignored when computing the result.
 * 
 * @param q SYCL queue.
 * @param A_buf Buffer containing the dense matrix elements in row-major order.
 * @param x_buf Buffer containing the vector elements to be multiplied with `A_buf`.
 * @param y_buf Buffer where the multiplication result is stored.
 * @param N Number of rows in the matrix `A_buf`.
 * @param M Number of columns in the matrix `A_buf` (and also the size of the vector `x_buf`).
 */
template<typename T>
void dense_matvec_multiplication(sycl::queue &q,
                                 sycl::buffer<T, 1> &A_buf,
                                 sycl::buffer<T, 1> &x_buf,
                                 sycl::buffer<T, 1> &y_buf,
                                 size_t N,
                                 size_t M) {
    q.submit([&](sycl::handler &h) {
         auto A_acc = A_buf.template get_access<sycl::access::mode::read>(h);
         auto x_acc = x_buf.template get_access<sycl::access::mode::read>(h);
         sycl::accessor y_acc(y_buf, h, sycl::write_only, sycl::no_init);
         h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
             size_t i = idx[0];
             T sum = static_cast<T>(0);
             for (size_t j = 0; j < M - 1; ++j) {
                 sum += A_acc[N * j + i] * x_acc[j];
             }
             y_acc[i] = sum;
         });
     }).wait();
}

/**
 * Perform dense matrix-vector multiplication where the input matrix and input vector are 1D device vectors, 
 * and the output vector is a 1D sycl::buffer.
 * Note: if the matrix has more rows than elements in the vector, then the
 * extra rows are ignored when computing the result.
 * 
 * @param q SYCL queue.
 * @param A Device vector containing the dense matrix elements in row-major order.
 * @param x Device vector containing the input vector
 * @param y_buf Buffer where the multiplication result is stored.
 * @param N Number of rows in the matrix `A_buf`.
 * @param M Number of columns in the matrix `A_buf` (and also the size of the vector `x_buf`).
 */
template<typename T>
void dense_matvec_multiplication(sycl::queue &q,
                                 T* A, 
                                 T* x,
                                 sycl::buffer<T, 1> &y_buf,
                                 size_t N,
                                 size_t M) {
    q.submit([&](sycl::handler &h) {
         sycl::accessor y_acc(y_buf, h, sycl::write_only, sycl::no_init);
         h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
             size_t i = idx[0];
             T sum = static_cast<T>(0);
             for (size_t j = 0; j < M - 1; ++j) {
                 sum += A[N * j + i] * x[j];
             }
             y_acc[i] = sum;
         });
     }).wait();
}

/**
 * @brief Computes the residual r = b - A*x of the linear system A*x = b, 
 * where `A`, `x`, and `b` are all sycl::buffers.
 *
 * @tparam T The data type of the elements in the matrix and vectors.
 * @param q1 SYCL queue.
 * @param A The CSR (Compressed Sparse Row) matrix representing the system matrix `A`.
 * @param x_buf SYCL buffer containing the solution vector `x`.
 * @param b_buf SYCL buffer containing the right-hand side vector `b`.
 * @param r_buf SYCL buffer where the residual vector `r` will be stored. The buffer must be pre-allocated with at least `n` elements.
 * @param Ax_buf SYCL buffer used to store the intermediate product `Ax`. This buffer must also be pre-allocated with at least `n` elements.
 * @param n The size of the vectors `x`, `b`, and `r`, and the number of rows in the matrix `A`.
 */
template<typename T>
void compute_residual(sycl::queue &q1, CSRMatrix<T>& A,
                      sycl::buffer<T, 1> &x_buf,
                      sycl::buffer<T, 1> &b_buf,
                      sycl::buffer<T, 1> &r_buf,
                      sycl::buffer<T, 1> &Ax_buf,
                      size_t n) {
    csr_matrix_vector_mult<T>(q1, A, x_buf, Ax_buf, n);
    q1.submit([&](sycl::handler &h) {
        sycl::accessor Ax_acc(Ax_buf, h, sycl::read_only);
        sycl::accessor b_acc(b_buf, h, sycl::read_only);
        sycl::accessor r_acc(r_buf, h, sycl::write_only, sycl::no_init);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            r_acc[i] = b_acc[i] - Ax_acc[i];
        });
    }).wait();
}

#endif