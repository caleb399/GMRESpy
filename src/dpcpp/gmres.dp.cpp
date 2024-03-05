#ifndef DPCPP_GMRES__GMRES_DP_HPP
#define DPCPP_GMRES__GMRES_DP_HPP

#include <iostream>
#include <string>

#include "csr.dp.hpp"          // CSR matrix struct definition
#include "gmres_buf.dp.hpp"    // Implementation using buffers (for CPU)
#include "gmres_device.dp.hpp" // Implementation using device vectors (for GPU)
#include "reductions.dp.hpp"   // For dot products / norms

// Solver parameters
struct GmresParams {
    double rtol = 1e-6;
    size_t restart = 40;
    size_t max_iter = 1000;
    std::string target_device = "DEFAULT";
    bool show_platform_info = true;
    bool show_progress = false;

    // Constructor w/ default values
    GmresParams(double rtol = 1e-6, size_t restart = 30, size_t max_iter = 1000, std::string target_device = "DEFAULT", bool show_platform_info = true, bool show_progress = false)
        : rtol(rtol), restart(restart), max_iter(max_iter), target_device(std::move(target_device)), show_platform_info(show_platform_info), show_progress(show_progress) {}
};

// Create queue based on the requested device.
sycl::queue create_queue(const std::string &target_device) {
    if (target_device == "CPU") {
        return sycl::queue(sycl::cpu_selector_v);
    } else if (target_device == "GPU") {
        return sycl::queue(sycl::gpu_selector_v);
    } else {
        return sycl::queue(sycl::default_selector_v);
    }
}

/**
 * Implements the Generalized Minimal Residual Method (GMRES) algorithm to
 * solve the linear system Ax = b.
 *
 * @tparam T The data type of the matrix and vectors; allowed values are float and double.
 *
 * @param A The system matrix, represented in compressed sparse row (CSR) format.
 * @param x0_buf A SYCL buffer containing the initial guess for the solution vector x.
 * @param b_buf A SYCL buffer representing the right-hand side vector of the linear system.
 * @param x A pointer to the memory where the solution vector will be stored. This array should be
 *          preallocated with at least `n` elements.
 * @param rerr Reference to a variable where the final relative error will be stored.
 * @param n Dimension of the system matrix A, i.e., A is an n x n matrix.
 * @param params (Optional) GmresParams struct containing solver parameters. If not provided, default values will be used.
 *
 * @return Returns -1 if not converged. If converged, returns the number of iterations performed.
 *
 * Note: This function modifies the input x buffer in-place with the computed solution and updates
 *       the rerr variable with the final relative error.
 */
template <typename T>
int64_t gmres(CSRMatrix<T> &A,
              sycl::buffer<T, 1> &x0_buf, // Initial guess
              sycl::buffer<T, 1> &b_buf,  // Right-hand side vector
              T *x,
              T &rerr,
              size_t n,
              GmresParams params = GmresParams()) {
    // Convergence parameters
    size_t restart = params.restart;
    size_t max_iter = params.max_iter;
    T rtol = static_cast<T>(params.rtol);
    bool show_progress = params.show_progress;

    // Convert target_device string to uppercase for case-insensitive comparison.
    std::string device_upper = params.target_device;
    std::transform(device_upper.begin(), device_upper.end(), device_upper.begin(), [](unsigned char c) { return std::toupper(c); });

    // Create the queue, trying to satisfy any request for the target device (CPU or GPU).
    sycl::queue q1 = create_queue(device_upper);
    if (params.show_platform_info) {
        std::cout << "Running GMRES on: " << q1.get_device().get_info<sycl::info::device::name>() << std::endl;
    }

    // Initialize buffers
    std::vector<T> r_vec(n);  // Residual vector.
    std::vector<T> Ax_vec(n); // For storing matrix-vector products.
    sycl::buffer<T> x_buf(x, sycl::range<1>(n));
    sycl::buffer<T> r_vec_buf(r_vec.data(), sycl::range<1>(n));
    sycl::buffer<T> Ax_buf(Ax_vec.data(), sycl::range<1>(n));

    // Compute the norm of vector b, to be used in convergence checks.
    std::vector<T> norm_b_vec(1, static_cast<T>(0));
    sycl::buffer<T, 1> norm_b_buf(norm_b_vec.data(), sycl::range<1>(norm_b_vec.size()));
    norm(q1, b_buf, norm_b_buf, n);
    sycl::host_accessor b_host_acc(norm_b_buf, sycl::read_only);
    T b_norm = b_host_acc[0];

    // Prepare for GMRES iterations.
    bool converged = false;
    int64_t iter = 0;
    std::vector<T> norm_r_vec(1, static_cast<T>(0));
    sycl::buffer<T, 1> norm_r_buf(norm_r_vec.data(), sycl::range<1>(norm_r_vec.size()));

    // Check if the queue is targeting a GPU, CPU, or something else
    // If GPU, then execute via `gmres_device_single`. If CPU or something else,
    // then execute via 'gmres_buf_single'. The former uses device vectors to speed up the
    // reduction operations involved in computing dot products; the latter uses buffers throughout
    // and is more flexible (though slower).
    if (q1.get_device().is_gpu()) {
        T *Q_dev = sycl::malloc_device<T>(n * (restart + 1), q1);

        for (iter = 0; iter < max_iter; ++iter) {
            // Do single iteration of GMRES and check for convergence
            rerr = gmres_device_single(q1, A, x0_buf, x_buf, b_buf, r_vec_buf, Ax_buf, Q_dev, b_norm, norm_r_buf, rtol, n, restart);
            if (show_progress) {
                std::cout << "Residual: " << rerr << std::endl;
            }
            if (rerr <= rtol) {
                converged = true;
                break;
            }

            // Update x0 for next iteration
            q1.submit([&](sycl::handler &h) {
                  auto x_acc = x_buf.template get_access<sycl::access::mode::read>(h);
                  auto x0_acc = x0_buf.template get_access<sycl::access::mode::write>(h);
                  h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                      size_t i = idx[0];
                      x0_acc[i] = x_acc[i];
                  });
              }).wait();
        }
        sycl::free(Q_dev, q1);
    } else {
        if (!q1.get_device().is_cpu()) {
            std::cout << "Executing on an untested device type. Proceed with caution." << std::endl;
        }

        std::vector<T> Q(n * (restart + 1), static_cast<T>(0));
        std::vector<T> H((restart + 1) * restart, static_cast<T>(0));
        sycl::buffer<T> Q_buf(Q.data(), sycl::range<1>((restart + 1) * n));
        sycl::buffer<T> H_buf(H.data(), sycl::range<1>((restart + 1) * restart));

        for (iter = 0; iter < max_iter; ++iter) {
            // Do single iteration of GMRES and check for convergence
            rerr = gmres_buf_single(q1, A, x0_buf, x_buf, b_buf, r_vec_buf, Ax_buf, Q_buf, H_buf, b_norm, norm_r_buf, rtol, n, restart);
            if (show_progress) {
                std::cout << "Residual: " << rerr << std::endl;
            }
            if (rerr <= rtol) {
                converged = true;
                break;
            }

            // Update x0 for next iteration
            q1.submit([&](sycl::handler &h) {
                  auto x_acc = x_buf.template get_access<sycl::access::mode::read>(h);
                  auto x0_acc = x0_buf.template get_access<sycl::access::mode::write>(h);
                  h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                      size_t i = idx[0];
                      x0_acc[i] = x_acc[i];
                  });
              }).wait();
        }
    }

    // Copy solution vector to host
    sycl::host_accessor x_host_acc(x_buf, sycl::read_only);
    for (int i = 0; i < n; i++) {
        x[i] = x_host_acc[i];
    }

    // If converged, return current iteration number; otherwise return -1.
    if (converged)
        return iter;
    return -1;
}

#endif