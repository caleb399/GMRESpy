/**
 * @file gmres_buf_dp.cpp
 * @brief Core routines for GMRES using sycl buffers. Suitable for execution on a CPU. 
 * @note Will also run on a device (e.g., GPU) but with significant performance degradations due
 * to inefficient memory transfers by the SYCl runtime.
*/

#include <vector>
#include "gmres_buf.dp.hpp"
#include "matrix_ops.dp.hpp"
#include "reductions.dp.hpp"


/**
 * Modified Gramd-schmidt process for GMRES.
 *
 * Orthogonalizes a set of vectors (columns of Q) up to the j-th column
 * and computes the Hessenberg matrix H. This is used in the Arnoldi iteration for GMRES.
 *
 * @param q The SYCL queue.
 * @param Q_buf sycl::buffer for the matrix whose columns are to be orthogonalized. On exit, Q contains the orthogonalized
 * vectors up to the j-th column.
 * @param H_buf sycl::buffer for the Hessenberg matrix H, dimension (m+1) x m
 * @param j Current Krylov subspace, i.e., we are orthogonalizing w.r.t. the last vector in the j^th Krylov subspace.
 * @param n Original problem size (number of rows in Q).
 * @param m Specifies Krylov subspace K_m (also number of columns in Q).
 */
template <typename T>
void gram_schmidt(sycl::queue &q1,
                  sycl::buffer<T> &Q_buf,
                  sycl::buffer<T> &H_buf,
                  size_t j,
                  size_t n,
                  size_t m) {

    // QQ_prod accumulates the results of dot product reduction in appropriate elements
    // of H, so we need to make sure those elements are set to zero before calling QQ_prod.
    q1.submit([&](sycl::handler &h) {
          sycl::accessor H_acc(H_buf, h, sycl::write_only, sycl::no_init);
          h.single_task([=]() { for (int k = 0; k < j; ++k) { H_acc[m * k + j - 1] = 0; } });
      }).wait();

    // Computes the dot products between Q[k,:] and Q[j,:], for k = 1 .. j - 1,
    // and stores the results in H[j-1,k].
    QQ_prod(q1, Q_buf, H_buf, n, j, m);
    q1.wait();

    // Q[j,:] -= dot(Q[:j,:], H[:j,j-1])
    q1.submit([&](sycl::handler &h) {
          auto Q_acc = Q_buf.template get_access<sycl::access::mode::read_write>(h);
          auto H_acc = H_buf.template get_access<sycl::access::mode::read>(h);
          h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
              size_t i = idx[0];
              T dp = static_cast<T>(0);
              for (int k = 0; k < j; ++k) {
                  dp += Q_acc[n * k + i] * H_acc[m * k + j - 1];
              }
              Q_acc[n * j + i] -= dp;
          });
      }).wait();

    // H[j, j - 1] = norm(Q[j,:]), Q[j, :] /= H[j, j - 1]
    auto x_vec = std::vector<T>(1);
    sycl::buffer<T, 1> x_buf(x_vec.data(), sycl::range<1>(x_vec.size()));
    dot_prod_offset(q1, Q_buf, Q_buf, x_buf, n, n * j, n * j);
    q1.wait();
    q1.submit([&](sycl::handler &h) {
        sycl::accessor H_acc(H_buf, h, sycl::write_only, sycl::no_init);
        sycl::accessor x_acc(x_buf, h, sycl::read_only);
        h.single_task([=]() { H_acc[j * m + j - 1] = sqrt(x_acc[0]); });
    });
    q1.submit([&](sycl::handler &h) {
        sycl::accessor Q_acc(Q_buf, h, sycl::read_write);
        sycl::accessor x_acc(x_buf, h, sycl::read_only);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            Q_acc[n * j + idx[0]] /= sqrt(x_acc[0]);
        });
    });
    q1.wait();
}

/**
 * Perform the Arnoldi iteration for computing an orthonormal basis of the Krylov subspace.
 *
 * @param q1 The SYCL queue.
 * @param A The CSRMatrix struct representing the sparse matrix `A`.
 * @param n The dimension of the matrix `A`.
 * @param m The number of Arnoldi steps to perform.
 * @param Q_buf The buffer where the orthonormal vectors (basis of the Krylov subspace) are stored.
 * @param H_buf The buffer where the upper Hessenberg matrix is stored.
 */
template <typename T>
void arnoldi_iteration(sycl::queue &q1, CSRMatrix<T> &A,
                       size_t n,
                       size_t m,
                       sycl::buffer<T, 1> &Q_buf,
                       sycl::buffer<T, 1> &H_buf) {
    for (int j = 1; j <= m; ++j) {
        csr_matrix_vector_mult<T>(q1, A, Q_buf, n, j - 1);
        gram_schmidt(q1, Q_buf, H_buf, j, n, m);
    }
}

/**
 * @brief Performs a single iteration of the GMRES algorithm, i.e.,
 * solves the corresponding linear system up to the `m` Krylov subspace `K_m`.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @param q1 The SYCL queue.
 * @param A The CSR (Compressed Sparse Row) matrix representing the system matrix `A`.
 * @param x0_buf SYCL buffercontaining the initial guess for the solution vector `x`.
 * @param x_buf SYCL buffer where the updated solution vector `x` will be stored.
 * @param b_buf SYCL buffer containing the right-hand side vector `b`.
 * @param r_buf SYCL buffer where the residual vector `r` (A*x - b) will be stored.
 * @param Ax_buf A temporary SYCL buffer used for storing intermediate matrix-vector products.
 * @param Q_buf SYCL buffer used for storing the orthonormal basis generated during the GMRES process.
 * @param H_buf SYCL buffer used for storing the Hessenberg matrix generated during the GMRES process.
 * @param b_norm The norm of the right-hand side vector `b`, used for convergence testing.
 * @param norm_r_buf SYCL buffer where the norm of the residual vector `r` will be stored.
 * @param rtol The relative tolerance for convergence.
 * @param n The dimension of the matrix `A` and the vectors `x`, `b`, and `r`.
 * @param m Solve system in the Krylov subspace `K_m`.
 */
template <typename T>
T gmres_buf_single(sycl::queue &q1, 
                   CSRMatrix<T> &A, 
                   sycl::buffer<T, 1> &x0_buf,
                   sycl::buffer<T, 1> &x_buf,
                   sycl::buffer<T, 1> &b_buf,
                   sycl::buffer<T, 1> &r_buf,
                   sycl::buffer<T, 1> &Ax_buf,
                   sycl::buffer<T, 1> &Q_buf,
                   sycl::buffer<T, 1> &H_buf,
                   T b_norm,
                   sycl::buffer<T, 1> &norm_r_buf,
                   const T rtol,
                   const size_t n,
                   const size_t m) {

    // Check residual
    compute_residual(q1, A, x0_buf, b_buf, r_buf, Ax_buf, n);
    norm(q1, r_buf, norm_r_buf, n);
    sycl::host_accessor r_host_acc(norm_r_buf, sycl::read_only);
    T residual = r_host_acc[0] / b_norm;
    if (residual < rtol) {
        return residual;
    }

    // Initialize first row of Q
    q1.submit([&](sycl::handler &h) {
          sycl::accessor r_acc(r_buf, h, sycl::read_only);
          sycl::accessor norm_r_acc(norm_r_buf, h, sycl::read_only);
          sycl::accessor Q_acc(Q_buf, h, sycl::write_only, sycl::no_init);
          h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
              size_t i = idx[0];
              Q_acc[i] = r_acc[i] / norm_r_acc[0];
          });
      }).wait();

    arnoldi_iteration(q1, A, n, m, Q_buf, H_buf);

    // Now solve least-square system in (m + 1) Krylov subspace.

    // Copy H from device to host and cast to double
    std::vector<double> H_out = std::vector<double>((m + 1) * m, 0);
    sycl::host_accessor H_host_acc(H_buf, sycl::read_only);
    for (size_t i = 0; i < m * (m + 1); i++) {
        H_out[i] = static_cast<double>(H_host_acc[i]);
    }

    // Initialize r.h.s. vector
    std::vector<double> b_out = std::vector<double>(m + 1, 0);
    sycl::host_accessor norm_r_host_acc(norm_r_buf, sycl::read_only);
    b_out[0] = norm_r_host_acc[0];

    // Solve system and send result to sycl buffer `y_buf`
    std::vector<double> y_dbl = solve_upper_hessenberg(H_out, b_out, m);
    std::vector<T> y = std::vector<T>(y_dbl.size());
    for (int i = 0; i < y_dbl.size(); i++) {
        y[i] = static_cast<T>(y_dbl[i]);
    }
    sycl::buffer<T, 1> y_buf(y.data(), sycl::range<1>(y.size()));

    // Construct solution vector in full n-dimensional space from least-squares sol.
    // in Krylov subspace
    dense_matvec_multiplication(q1, Q_buf, y_buf, x_buf, n, (m + 1));
    q1.submit([&](sycl::handler &h) {
          sycl::accessor x_acc(x_buf, h, sycl::read_write);
          sycl::accessor x0_acc(x0_buf, h, sycl::read_only);
          h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
              size_t i = idx[0];
              x_acc[i] = x0_acc[i] + x_acc[i];
          });
      }).wait();

    // Residual vector
    compute_residual(q1, A, x_buf, b_buf, r_buf, Ax_buf, n);
    auto res_vec = std::vector<T>(1, 0);
    sycl::buffer<T, 1> res_buf(res_vec.data(), sycl::range<1>(res_vec.size()));
    norm(q1, r_buf, res_buf, n);
    sycl::host_accessor res_host_acc(res_buf, sycl::read_only);

    residual = res_host_acc[0] / b_norm;
    return residual;
}

// Explicit template instantiation
template float gmres_buf_single<float>(sycl::queue &q1, CSRMatrix<float> &A, sycl::buffer<float, 1> &x0_buf, sycl::buffer<float, 1> &x_buf, sycl::buffer<float, 1> &b_buf, sycl::buffer<float, 1> &r_buf, sycl::buffer<float, 1> &Ax_buf, sycl::buffer<float, 1> &Q_buf, sycl::buffer<float, 1> &H_buf, float b_norm, sycl::buffer<float, 1> &norm_r_buf, const float rtol, const size_t n, const size_t m);

template double gmres_buf_single<double>(sycl::queue &q1, CSRMatrix<double> &A, sycl::buffer<double, 1> &x0_buf, sycl::buffer<double, 1> &x_buf, sycl::buffer<double, 1> &b_buf, sycl::buffer<double, 1> &r_buf, sycl::buffer<double, 1> &Ax_buf, sycl::buffer<double, 1> &Q_buf, sycl::buffer<double, 1> &H_buf, double b_norm, sycl::buffer<double, 1> &norm_r_buf, const double rtol, const size_t n, const size_t m);


/*
 * The following function `QQ_prod` contains a modified version of code from 
 * https://github.com/rvperi/DPCPP-Reduction/blob/main/pum.cpp
 *
 * Original Copyright (c) 2021 Ramesh Peri
 *
 * Modifications (03/04/2024):
 * - Changed `multiBlockInterleavedReduction` to compute the dot product of two vectors
 * - Other small changes
 *
 * The original source code is licensed under the MIT License. See below for the full license text of the original source.
 */

/*
MIT License

Copyright (c) 2021 Ramesh Peri

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/**
 * Calculates dot products between rows of an M x N matrix (matrix Q) represented by a SYCL buffer.
 * Specifically, dot products are calculated between the j^th row and each of the first (j-1)
 * rows (j-1 dot products in total). For use with the modified Gram-Schmidt algorithm, the dot products are stored
 * in the first j elements of the (j-1)th column of a separate, (m+1) x m matrix (matrix H), though the output buffer indexing
 * can easily be changed for other applications.
 *
 * @tparam T The data type of the matrix elements. Must support atomic operations.
 * @param q The SYCL queue.
 * @param Q_buf SYCL buffer containing the input matrix (matrix Q).
 * @param C_buf SYCL buffer containing matrix C, where the result of the dot product for each row will be stored.
 * @param n Row dimension of matrix Q
 * @param j Compute dot products up to the j^th row of Q.
 * @param m Column dimension of matrix C.
 */
template <typename T>
void QQ_prod(sycl::queue &q,
             sycl::buffer<T> Q_buf,
             sycl::buffer<T> &H_buf,
             size_t n,
             int j,
             size_t m) {
    const int work_group_size = q.get_device().get_info<sycl::info::device::max_work_group_size>();
    const int num_work_items = n / sizeof(T);
    const size_t start_index = n * j; // (Linear) index within Q from where to start the dot product calculation.
    q.submit([&](auto &h) {
        const sycl::accessor Q_acc(Q_buf, h);
        sycl::accessor H_acc(H_buf, h, sycl::write_only, sycl::no_init);
        sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local> scratch(j, h);
        sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local> dp(j, h);
        h.parallel_for(sycl::nd_range<1>{sycl::range<1>(num_work_items), sycl::range<1>(work_group_size)},
            [=](sycl::nd_item<1> item) {
                size_t glob_id = item.get_global_id(0);
                size_t loc_id = item.get_local_id(0);
                if (loc_id < j)
                    scratch[loc_id] = static_cast<T>(0);
                sycl::vec<T, 4> A_val, B_val; // We will load 4 elements each of the input rows.
                using global_ptr = sycl::multi_ptr<T, sycl::access::address_space::global_space>;
                for (int k = 0; k < j; ++k) {
                    A_val.load(glob_id, global_ptr(&Q_acc[start_index]));
                    B_val.load(glob_id, global_ptr(&Q_acc[n * k])); // k^th row of B
                    T dp = A_val[0] * B_val[0] + A_val[1] * B_val[1] + A_val[2] * B_val[2] + A_val[3] * B_val[3];
                    item.barrier(sycl::access::fence_space::local_space);
                    auto vl = sycl::atomic_ref<T,
                                                sycl::memory_order::relaxed,
                                                sycl::memory_scope::work_group,
                                                sycl::access::address_space::local_space>(
                        scratch[k]);
                    vl.fetch_add(dp);
                }
                item.barrier(sycl::access::fence_space::local_space);
                for (int k = 0; k < j; ++k) {
                    if (loc_id == 0) {
                        auto v = sycl::atomic_ref<T,
                                                    sycl::memory_order::relaxed,
                                                    sycl::memory_scope::device,
                                                    sycl::access::address_space::global_space>(
                            H_acc[m * k + j - 1]);
                        v.fetch_add(scratch[k]);
                    }
                }
        });
    });
}