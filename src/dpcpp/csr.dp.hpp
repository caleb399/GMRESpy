/**
 * @file csr.dp.hpp
 * @brief Definition of CSRMatrix structure for sparse CSR matrices.
*/

#ifndef DPCPP_GMRES__CSR_DP_HPP
#define DPCPP_GMRES__CSR_DP_HPP

#include <CL/sycl.hpp>

template <typename T>
struct CSRMatrix {
    sycl::buffer<T, 1> values_buf;
    sycl::buffer<int32_t, 1> col_indices_buf;
    sycl::buffer<int32_t, 1> row_offsets_buf;

    CSRMatrix(sycl::buffer<T, 1> values,
              sycl::buffer<int32_t, 1> col_indices,
              sycl::buffer<int32_t, 1> row_offsets)
        : values_buf(std::move(values)),
          col_indices_buf(std::move(col_indices)),
          row_offsets_buf(std::move(row_offsets)) {}
};

#endif