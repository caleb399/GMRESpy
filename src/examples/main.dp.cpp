#include "../dpcpp/gmres.dp.hpp"
#include "../hdf5/hdf5_util.hpp"

const std::string datadir = "test_data"; // Set to location of test data directory, containing "test1.h5"

int main() {

    // Queues
    sycl::queue q1(sycl::gpu_selector_v);
    std::cout << "Selected Device: " << q1.get_device().get_info<sycl::info::device::name>() << std::endl;

    // Input files
    std::string datafilename = datadir + "test1.h5";

    // Matrix
    std::string datasetRowPtr = "indptr";
    std::string datasetColInd = "indices";
    std::string datasetVal = "data";
    std::vector<int32_t> row_offsets;
    std::vector<int32_t> col_indices;
    std::vector<float> values;
    size_t n, cols;
    if (readSparseMatrixFromHDF5(datafilename, datasetRowPtr, datasetColInd, datasetVal, row_offsets, col_indices, values, n, cols)) {
        std::cout << "Sparse matrix successfully read from HDF5 file." << std::endl;
        std::cout << "rows: " << n << std::endl;
        std::cout << "cols: " << cols << std::endl;
    } else {
        std::cerr << "Failed to read the matrix." << std::endl;
    }

    // Vectors
    std::vector<float> x(n, static_cast<float>(0));
    auto x0 = x;
    std::vector<float> b;
    read1DArray(datafilename, "b", b);

    // Buffer initialization
    sycl::buffer<float, 1> values_buf(values.data(), sycl::range<1>(values.size()));
    sycl::buffer<int32_t, 1> col_indices_buf(col_indices.data(), sycl::range<1>(col_indices.size()));
    sycl::buffer<int32_t, 1> row_offsets_buf(row_offsets.data(), sycl::range<1>(row_offsets.size()));
    sycl::buffer<float, 1> x0_buf(x0.data(), sycl::range<1>(x0.size()));
    sycl::buffer<float, 1> b_buf(b.data(), sycl::range<1>(b.size()));
    CSRMatrix<float> A(values_buf, col_indices_buf, row_offsets_buf);

    // Solver parameters
    GmresParams params;
    params.rtol = 1e-3;

    // Run solver
    auto start = std::chrono::high_resolution_clock::now();
    float rerr;
    auto status = gmres<float>(A,
                x0_buf,
                b_buf,
                x.data(),
                rerr, n, params);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "    Runtime: " << elapsed.count() << " seconds" << std::endl;

    if (status > 0)
        std::cout << "GMRES converged after " << status << " iterations with relative error of " << rerr << "." << std::endl;
    else if (status == -1) {
        std::cout << "GMRES reached the maximum number of iterations (relative error: " << rerr << ")." << std::endl;
    }
    else {
        std::cout << "GMRES exited without performing any iterations (relative error: " << rerr << ")." << std::endl;
    }

    return 0;
}