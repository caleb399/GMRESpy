/**
 * @file gmres_device_dp.cpp
 * @brief Core routines for GMRES on a SYCL device (e.g., GPU)
*/

#include <vector>
#include "gmres_device.dp.hpp"
#include "matrix_ops.dp.hpp"
#include "reductions.dp.hpp"

/**
 * Modified Gramd-schmidt process for GMRES.
 *
 * Orthogonalizes a set of vectors (columns of Q) up to the j-th column
 * and computes the Hessenberg matrix H. This is used in the Arnoldi iteration for GMRES.
 *
 * @param q The SYCL queue.
 * @param Q_dev Device vector for the matrix whose columns are to be orthogonalized. On exit, Q contains the orthogonalized
 * vectors up to the j-th column.
 * @param H_dev Device vector for the Hessenberg matrix H, dimension (m+1) x m
 * @param j Current Krylov subspace, i.e., we are orthogonalizing w.r.t. the last vector in the j^th Krylov subspace.
 * @param n Original problem size (number of rows in Q).
 * @param m Specifies Krylov subspace K_m (also number of columns in Q).
 */
template <typename T>
void gram_schmidt(sycl::queue &q1,
                  T *Q_dev,
                  T *H_dev,
                  size_t j,
                  size_t n,
                  size_t m) {

    // Computes the dot products between Q[k,:] and Q[j,:], for k = 1 .. j - 1,
    // and stores the results in H[j-1,k].
    for (int k = 0; k < j; ++k) {
        dot_prod_offset(q1, Q_dev, Q_dev, &H_dev[m * k + j - 1], n, n * j, n * k);
    }
    q1.wait();

    // Q[j,:] -= dot(Q[:j,:], H[:j,j-1])
    q1.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            size_t i = idx[0];
            T dp = static_cast<T>(0);
            for (int k = 0; k < j; ++k) {
                dp += Q_dev[n * k + i] * H_dev[m * k + j - 1];
            }
            Q_dev[n * j + i] -= dp;
        });
    }).wait();

    // H[j, j - 1] = norm(Q[j,:]), Q[j, :] /= H[j, j - 1]
    T *H_val = sycl::malloc_device<T>(1, q1);
    dot_prod_offset(q1, Q_dev, Q_dev, &H_val[0], n, n * j, n * j);
    q1.wait();
    q1.submit([&](sycl::handler &h) {
        h.single_task([=]() { H_dev[j * m + j - 1] = sqrt(H_val[0]); });
    });
    q1.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            size_t i = idx[0];
            Q_dev[n * j + i] /= sqrt(H_val[0]);
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
 * @param Q_dev The device vector where the orthonormal vectors (basis of the Krylov subspace) are stored.
 * @param H_dev The device vector where the upper Hessenberg matrix is stored.
 */
template <typename T>
void arnoldi_iteration(sycl::queue &q1, CSRMatrix<T> &A,
                       size_t n,   // Dimension of the matrix
                       size_t m,   // Number of Arnoldi steps
                       T *Q_dev,   // Orthogonal vectors
                       T *H_dev) { // Hessenberg matrix
    for (int j = 1; j <= m; ++j) {
        csr_matrix_vector_mult<T>(q1, A, Q_dev, n, j - 1);
        gram_schmidt(q1, Q_dev, H_dev, j, n, m);
    }
}

/**
 * @brief Performs a single iteration of GMRES, i.e.,
 * solves the corresponding linear system up to the `m` Krylov subspace `K_m`.
 *
 * @tparam T The data type of the matrix and vector elements.
 * @param q1 The SYCL queue.
 * @param A The CSR (Compressed Sparse Row) matrix representing the system matrix `A`.
 * @param x0_buf SYCL buffercontaining the initial guess for the solution vector `x`.
 * @param x_buf SYCL bufferwhere the updated solution vector `x` will be stored.
 * @param b_buf SYCL buffercontaining the right-hand side vector `b`.
 * @param r_buf SYCL bufferwhere the residual vector `r` (A*x - b) will be stored.
 * @param Ax_buf Temporary SYCL buffer used for storing intermediate matrix-vector products.
 * @param Q_dev Device vector used for storing the orthonormal basis generated during the GMRES process.
 * @param b_norm The norm of the right-hand side vector `b`, used for convergence testing.
 * @param norm_r_buf SYCL bufferwhere the norm of the residual vector `r` will be stored.
 * @param rtol The relative tolerance for convergence.
 * @param n The dimension of the matrix `A` and the vectors `x`, `b`, and `r`.
 * @param m Solve system in the Krylov subspace `K_m`.
 */
template <typename T>
T gmres_device_single(sycl::queue &q1, 
                      CSRMatrix<T> &A, 
                      sycl::buffer<T, 1> &x0_buf,
                      sycl::buffer<T, 1> &x_buf,
                      sycl::buffer<T, 1> &b_buf,
                      sycl::buffer<T, 1> &r_buf,
                      sycl::buffer<T, 1> &Ax_buf,
                      T *Q_dev,
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

    T *H_dev = sycl::malloc_device<T>(m * (m + 1), q1);

    // Initialize first row of Q
    q1.submit([&](sycl::handler &h) {
          auto r_acc = r_buf.template get_access<sycl::access::mode::read>(h);
          auto norm_r_acc = norm_r_buf.template get_access<sycl::access::mode::read>(h);
          h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
              size_t i = idx[0];
              Q_dev[i] = r_acc[i] / norm_r_acc[0];
          });
      }).wait();

    arnoldi_iteration(q1, A, n, m, Q_dev, H_dev);

    // Now solve least-square system in (m + 1) Krylov subspace.

    // Copy H from device to host
    std::vector<T> H_out = std::vector<T>((m + 1) * m, 0);
    q1.memcpy(H_out.data(), H_dev, (m + 1) * m * sizeof(T)).wait();

    // Initialize r.h.s.
    std::vector<double> b_out = std::vector<double>(m + 1, static_cast<double>(0));
    sycl::host_accessor norm_r_host_acc(norm_r_buf, sycl::read_only);
    b_out[0] = norm_r_host_acc[0];

    // Convert H to double precision
    std::vector<double> H_dbl((m + 1) * m);
    std::transform(H_out.begin(), H_out.end(), H_dbl.begin(), [](T val) -> double {
        return static_cast<double>(val);
    });

    // Solve system and process result
    std::vector<double> y_dbl = solve_upper_hessenberg(H_dbl, b_out, m);
    std::vector<T> y(y_dbl.size());
    for (size_t i = 0; i < y_dbl.size(); i++) {
        y[i] = static_cast<T>(y_dbl[i]);
    }
    T *y_dev = sycl::malloc_device<T>(m, q1);
    q1.memcpy(y_dev, y.data(), m * sizeof(T)).wait();

    // Construct solution vector in full n-dimensional space from least-squares sol.
    // in Krylov subspace
    dense_matvec_multiplication(q1, Q_dev, y_dev, x_buf, n, (m + 1));
    q1.submit([&](sycl::handler &h) {
        auto x_acc = x_buf.template get_access<sycl::access::mode::read_write>(h);
        auto x0_acc = x0_buf.template get_access<sycl::access::mode::read>(h);
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

    sycl::free(y_dev, q1);
    sycl::free(H_dev, q1);

    residual = res_host_acc[0] / b_norm;
    return residual;
}

template float gmres_device_single<float>(sycl::queue &q1, CSRMatrix<float> &A, sycl::buffer<float, 1> &x0_buf, sycl::buffer<float, 1> &x_buf, sycl::buffer<float, 1> &b_buf, sycl::buffer<float, 1> &r_buf, sycl::buffer<float, 1> &Ax_buf, float *Q_dev, float b_norm, sycl::buffer<float, 1> &norm_r_buf, const float rtol, const size_t n, const size_t m);

template double gmres_device_single<double>(sycl::queue &q1, CSRMatrix<double> &A, sycl::buffer<double, 1> &x0_buf, sycl::buffer<double, 1> &x_buf, sycl::buffer<double, 1> &b_buf, sycl::buffer<double, 1> &r_buf, sycl::buffer<double, 1> &Ax_buf, double *Q_dev, double b_norm, sycl::buffer<double, 1> &norm_r_buf, const double rtol, const size_t n, const size_t m);