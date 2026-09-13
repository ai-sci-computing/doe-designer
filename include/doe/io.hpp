/// @file io.hpp
/// @brief Data files: NumPy `.npy`, a small JSON writer, CSV and SVG line plots.
#pragma once

#include "doe/array.hpp"

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace doe {

/// Write a 2-D array as `.npy` (format version 1.0: magic, header dict with
/// `descr` `<f8`/`<f4`, `fortran_order: False`, `shape: (rows, cols)`, padded
/// so the data starts at a multiple of 64 bytes), C order = our row-major
/// layout, so `numpy.load` returns an array `a[i, j]` with i along x.
template <class T>
void write_npy(const std::filesystem::path& path, const Array2<T>& a);

/// Read a `.npy` written by write_npy() (or NumPy: little-endian float32 /
/// float64, C order, 2-D). Always returns double.
Array2<double> read_npy(const std::filesystem::path& path);

/// Minimal JSON object builder with insertion-ordered keys; numbers are
/// written with shortest round-trip precision, NaN/inf as `null`.
class Json {
public:
    /// Add a string member.
    void set(const std::string& key, const std::string& value);
    /// Add a string member.
    void set(const std::string& key, const char* value) { set(key, std::string(value)); }
    /// Add a number member (NaN/inf become null).
    void set(const std::string& key, double value);
    /// Add an integer member.
    void set(const std::string& key, int value);
    /// Add a boolean member.
    void set(const std::string& key, bool value);
    /// Add a nested object.
    void set(const std::string& key, const Json& value);
    /// Add an array of numbers.
    void set(const std::string& key, const std::vector<double>& values);
    /// Add an array of strings.
    void set(const std::string& key, const std::vector<std::string>& values);
    /// Serialize with two-space indentation.
    std::string dump(int indent = 0) const;

private:
    std::vector<std::pair<std::string, std::string>> items_;  ///< key -> serialized value
};

/// Write a string to a file (overwrites).
void write_text(const std::filesystem::path& path, const std::string& text);

/// Write rows of numbers with a header line.
void write_csv(const std::filesystem::path& path, const std::vector<std::string>& columns,
               const std::vector<std::vector<double>>& rows);

/// Simple SVG line plot: axes, tick labels, legend, one polyline per series.
struct SvgPlot {
    std::string title;    ///< plot title
    std::string x_label;  ///< x axis label
    std::string y_label;  ///< y axis label
    bool log_y = false;   ///< logarithmic y axis
    int width = 720;      ///< image width in px
    int height = 420;     ///< image height in px
    /// Add a series sampled at x = 0, 1, 2, ... (e.g. an energy history).
    /// `dashed` draws it with a dash pattern; `color` picks a palette index
    /// (-1: the next unused one), so related series can share a color.
    void add_series(const std::string& name, const std::vector<double>& y, bool dashed = false, int color = -1);
    /// Add a series with explicit x values.
    void add_series(const std::string& name, const std::vector<double>& x, const std::vector<double>& y,
                    bool dashed = false, int color = -1);
    /// The SVG document as text.
    std::string render() const;

private:
    struct Series {
        std::string name;
        std::vector<double> x, y;
        bool dashed = false;
        int color = 0;
    };
    std::vector<Series> series_;
};

/// @cond INTERNAL
extern template void write_npy<float>(const std::filesystem::path&, const Array2<float>&);
extern template void write_npy<double>(const std::filesystem::path&, const Array2<double>&);
/// @endcond

}  // namespace doe
