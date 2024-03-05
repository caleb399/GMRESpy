#include <iostream>
#include <vector>
#include <H5Cpp.h>

bool readSparseMatrixFromHDF5(const std::string &filename, 
                              const std::string &datasetRowPtr, 
                              const std::string &datasetColInd, 
                              const std::string &datasetVal, 
                              std::vector<int32_t> &csrRowPtr, 
                              std::vector<int32_t> &csrColInd, 
                              std::vector<float> &csrVal, 
                              size_t& rows, size_t& cols) {
    try {
        H5::H5File file(filename, H5F_ACC_RDONLY);

        // Read matrix dimensions
        {
            H5::DataSet dataset = file.openDataSet("shape");
            H5::DataSpace dataspace = dataset.getSpace();

            if (dataspace.getSimpleExtentNdims() != 1) {
                std::cerr << "shape dataset is not one-dimensional." << std::endl;
                return false;
            }

            hsize_t dims[1];
            dataspace.getSimpleExtentDims(dims, nullptr);
            if (dims[0] != 2) {
                std::cerr << "shape array does not have two elements." << std::endl;
                return false;
            }

            int size[2];
            dataset.read(size, H5::PredType::NATIVE_INT);
            rows = static_cast<size_t>(size[0]);
            cols = static_cast<size_t>(size[1]);
        }

        // Read csrRowPtr
        {
            H5::DataSet dataset = file.openDataSet(datasetRowPtr);
            H5::DataSpace dataspace = dataset.getSpace();

            if (dataspace.getSimpleExtentNdims() != 1) {
                std::cerr << "Dataset " << datasetRowPtr << " is not one-dimensional." << std::endl;
                return false;
            }

            hsize_t dims[1];
            dataspace.getSimpleExtentDims(dims, nullptr);
            csrRowPtr.resize(dims[0]);
            dataset.read(csrRowPtr.data(), H5::PredType::NATIVE_INT);
        }

        // Read csrColInd
        {
            H5::DataSet dataset = file.openDataSet(datasetColInd);
            H5::DataSpace dataspace = dataset.getSpace();

            if (dataspace.getSimpleExtentNdims() != 1) {
                std::cerr << "Dataset " << datasetColInd << " is not one-dimensional." << std::endl;
                return false;
            }

            hsize_t dims[1];
            dataspace.getSimpleExtentDims(dims, nullptr);
            csrColInd.resize(dims[0]);
            dataset.read(csrColInd.data(), H5::PredType::NATIVE_INT);
        }

        // Read csrVal
        {
            H5::DataSet dataset = file.openDataSet(datasetVal);
            H5::DataSpace dataspace = dataset.getSpace();

            if (dataspace.getSimpleExtentNdims() != 1) {
                std::cerr << "Dataset " << datasetVal << " is not one-dimensional." << std::endl;
                return false;
            }

            hsize_t dims[1];
            dataspace.getSimpleExtentDims(dims, nullptr);
            csrVal.resize(dims[0]);
            dataset.read(csrVal.data(), H5::PredType::NATIVE_FLOAT);
        }

        return true;
    } catch (const H5::FileIException &e) {
        std::cerr << "File I/O error: " << e.getCDetailMsg() << std::endl;
    } catch (const H5::DataSetIException &e) {
        std::cerr << "Dataset I/O error: " << e.getCDetailMsg() << std::endl;
    } catch (const H5::DataSpaceIException &e) {
        std::cerr << "Dataspace I/O error: " << e.getCDetailMsg() << std::endl;
    } catch (const H5::DataTypeIException &e) {
        std::cerr << "Data type error: " << e.getCDetailMsg() << std::endl;
    } catch (...) {
        std::cerr << "An unknown error occurred." << std::endl;
    }
    return false;
}


bool read1DArray(const std::string &filename, const std::string &datasetName, std::vector<float> &rhs) {
    try {
        H5::H5File file(filename, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet(datasetName);
        H5::DataSpace dataspace = dataset.getSpace();

        // Check the number of dimensions in the dataset.
        if (dataspace.getSimpleExtentNdims() != 1) {
            std::cerr << "Dataset " << datasetName << " is not a one-dimensional array." << std::endl;
            return false;
        }

        // Get the size of the array.
        hsize_t dims[1];
        dataspace.getSimpleExtentDims(dims, nullptr);
        rhs.resize(dims[0]);

        // Read the data into the rhs vector.
        dataset.read(rhs.data(), H5::PredType::NATIVE_FLOAT);
        return true;
    } catch (const H5::FileIException &e) {
        std::cerr << "File I/O error: " << e.getCDetailMsg() << std::endl;
    } catch (const H5::DataSetIException &e) {
        std::cerr << "Dataset I/O error: " << e.getCDetailMsg() << std::endl;
    } catch (const H5::DataSpaceIException &e) {
        std::cerr << "Dataspace I/O error: " << e.getCDetailMsg() << std::endl;
    } catch (const H5::DataTypeIException &e) {
        std::cerr << "Data type error: " << e.getCDetailMsg() << std::endl;
    } catch (...) {
        std::cerr << "An unknown error occurred." << std::endl;
    }
    return false;
}
