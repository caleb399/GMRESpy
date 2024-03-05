/**
 * @file reductions.dp.hpp
 * @brief Various overloads of parallel dot product reduction.
*/

#ifndef DPCPP_GMRES__REDUCTIONS_DP_HPP
#define DPCPP_GMRES__REDUCTIONS_DP_HPP

#include <CL/sycl.hpp>

/**
 * l^2 norm of a vector
 *
 * @param q SYCl queue.
 * @param x_buf The input buffer containing the vector elements.
 * @param norm_result_buf The output buffer to store the norm result.
 * @param n The number of elements in the input vector.
 */
template<typename T>
void norm(sycl::queue &q,
          sycl::buffer<T, 1> &x_buf,
          sycl::buffer<T, 1> &norm_result_buf,
          size_t n) {
    dot_prod(q, x_buf, x_buf, norm_result_buf, n);
    q.submit([&](sycl::handler &h) {
        auto acc_norm_r = norm_result_buf.template get_access<sycl::access::mode::read_write>(h);
        h.single_task([=]() { acc_norm_r[0] = sqrt(acc_norm_r[0]); });
    }).wait();
}

/**
 * Regular dot product of two vectors
 *
 * @param q SYCl queue.
 * @param a_buf The first input buffer containing vector elements.
 * @param b_buf The second input buffer containing vector elements.
 * @param dp_buf The output buffer to store the dot product result.
 * @param n The number of elements in each input vector.
 */
template<typename T>
void dot_prod(sycl::queue &q,
              sycl::buffer<T, 1> &a_buf,
              sycl::buffer<T, 1> &b_buf,
              sycl::buffer<T, 1> &dp_buf,
              size_t n) {
    auto prop_list = sycl::property_list{sycl::property::reduction::initialize_to_identity()};
    q.submit([&](sycl::handler &h) {
        auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
        auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
        auto dot_product_reduction = reduction(dp_buf, h, std::plus<>(), prop_list);
        h.parallel_for(n, dot_product_reduction, [=](sycl::id<1> i, auto &reduction) {
            reduction.combine(a_acc[i] * b_acc[i]);
        });
    }).wait();
}

/**
 * Perform the dot product of two vectors with offset start indices. Overload in which the result
 * is stored in a SYCL buffer
 *
 * @param q SYCl queue.
 * @param a_buf Buffer for first input vector.
 * @param b_buf Buffer for second input vector.
 * @param dp_buf The single element buffer to store the dot product result.
 * @param n Number of elements in input vectors.
 * @param offset_a The offset for the first vector elements.
 * @param offset_b The offset for the second vector elements.
 */
template<typename T>
void dot_prod_offset(sycl::queue &q,
                      sycl::buffer<T> &a_buf,
                      sycl::buffer<T> &b_buf,
                      sycl::buffer<T> &dp_buf,
                      size_t n,
                      int64_t offset_a,
                      int64_t offset_b) {
    auto prop_list = sycl::property_list{sycl::property::reduction::initialize_to_identity()};
    q.submit([&](sycl::handler &h) {
        auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
        auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
        auto dot_product_reduction = reduction(dp_buf, h, std::plus<>(), prop_list);
        h.parallel_for(n, dot_product_reduction, [=](sycl::id<1> i, auto &reduction) {
            reduction.combine(a_acc[i + offset_a] * b_acc[i + offset_b]);
        });
    });
}

/**
 * Perform the dot product of two vectors with offset start indices. Overload in which the result
 * is stored in a device vector.
 *
 * @param q SYCl queue.
 * @param a_buf Buffer for first input vector.
 * @param b_buf Buffer for second input vector.
 * @param dp_dev The single-element device vector for storing the result.
 * @param n Number of elements in input vectors.
 * @param offset_a The offset for the first vector elements.
 * @param offset_b The offset for the second vector elements.
 */
template<typename T>
void dot_prod_offset(sycl::queue &q,
                      sycl::buffer<T> &a_buf,
                      sycl::buffer<T> &b_buf,
                      T* dp_dev,
                      size_t n,
                      int64_t offset_a,
                      int64_t offset_b) {
    auto prop_list = sycl::property_list{sycl::property::reduction::initialize_to_identity()};
    q.submit([&](sycl::handler &h) {
        auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
        auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
        auto dot_product_reduction = reduction(dp_dev, std::plus<>(), prop_list);
        h.parallel_for(n, dot_product_reduction, [=](sycl::id<1> i, auto &reduction) {
            reduction.combine(a_acc[i + offset_a] * b_acc[i + offset_b]);
        });
    });
}

/**
 * Perform the dot product of two vectors with offset start indices. Overload in which input vectors
 * and output are all device vectors.
 *
 * @param q SYCl queue.
 * @param a_dev Buffer for first input vector.
 * @param b_dev Buffer for second input vector.
 * @param dp_dev The single-element device vector for storing the result.
 * @param n Number of elements in input vectors.
 * @param offset_a The offset for the first vector elements.
 * @param offset_b The offset for the second vector elements.
 */
template<typename T>
void dot_prod_offset(sycl::queue &q,
                      T* a_dev,
                      T* b_dev,
                      T* dp_dev,
                      size_t n,
                      int64_t offset_a,
                      int64_t offset_b) {
    auto prop_list = sycl::property_list{sycl::property::reduction::initialize_to_identity()};
    q.submit([&](sycl::handler &h) {
        auto dot_product_reduction = reduction(dp_dev, std::plus<>(), prop_list);
        h.parallel_for(n, dot_product_reduction, [=](sycl::id<1> i, auto &reduction) {
            reduction.combine(a_dev[i + offset_a] * b_dev[i + offset_b]);
        });
    });
}

#endif