#include "H5Cpp.h"
#include <vector>

bool readSparseMatrixFromHDF5(const std::string& filename,
    const std::string& datasetRowPtr,
    const std::string& datasetColInd,
    const std::string& datasetVal,
    std::vector<int32_t>& csrRowPtr,
    std::vector<int32_t>& csrColInd,
    std::vector<float>& csrVal,
    size_t& rows, size_t& cols);

bool read1DArray(const std::string &filename, 
    const std::string &datasetName, 
    std::vector<float> &rhs);