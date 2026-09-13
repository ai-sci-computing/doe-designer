/// @file array.hpp
/// @brief Minimal dense 2-D array used for fields, images and masks.
#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace doe {

/// Row-major 2-D array. Index (i, j) = (x sample, y sample), i.e. the first
/// index runs along x exactly like NumPy's `meshgrid(..., indexing="ij")` in
/// the reference implementation, so that array layouts can be compared 1:1.
template <class T>
struct Array2 {
    std::size_t rows = 0;  ///< number of samples along the first index (x)
    std::size_t cols = 0;  ///< number of samples along the second index (y)
    std::vector<T> data;   ///< rows * cols elements, row-major

    Array2() = default;
    /// Allocate r x c elements initialized to `value`.
    Array2(std::size_t r, std::size_t c, T value = T{}) : rows(r), cols(c), data(r * c, value) {}

    /// Element (i, j), i along x.
    T& operator()(std::size_t i, std::size_t j) { return data[i * cols + j]; }
    /// Element (i, j), i along x.
    const T& operator()(std::size_t i, std::size_t j) const { return data[i * cols + j]; }

    /// Total number of elements.
    std::size_t size() const { return data.size(); }
    /// True when both arrays have the same rows and cols.
    bool same_shape(const Array2& other) const { return rows == other.rows && cols == other.cols; }
};

/// Complex scalar field sampled on a grid, in the working precision T.
template <class T>
using Field = Array2<std::complex<T>>;

}  // namespace doe
