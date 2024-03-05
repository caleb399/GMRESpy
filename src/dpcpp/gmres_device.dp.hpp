/**
 * @file gmres_device.dp.hpp
 * @brief Template header for the function `gmres_device_single` defined in file gmres_device.dp.cpp
*/

#ifndef DPCPP_GMRES__GMRES_DEVICE_DP_HPP
#define DPCPP_GMRES__GMRES_DEVICE_DP_HPP

#include <CL/sycl.hpp>
#include "csr.dp.hpp"

template<typename T>
T gmres_device_single(sycl::queue &q1, CSRMatrix<T>& A,
                        sycl::buffer<T, 1> &x0_buf,
                        sycl::buffer<T, 1> &x_buf,
                        sycl::buffer<T, 1> &b_buf,
                        sycl::buffer<T, 1> &r_buf,
                        sycl::buffer<T, 1> &Ax_buf,
                        T* Q_dev,
                        T b_norm,
                        sycl::buffer<T, 1> &norm_r_buf,
                        const T rtol,
                        const size_t n,
                        const size_t m);

#endif