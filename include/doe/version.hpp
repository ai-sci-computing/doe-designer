/// @file version.hpp
/// @brief Library version string and the versions of the numerical back ends.
#pragma once

#include <string>

namespace doe {

/// Semantic version of the doe-designer library.
std::string version();

/// Version string reported by the linked FFTW library (double precision build).
std::string fftw_version_string();

/// Version string reported by the linked libpng.
std::string libpng_version_string();

}  // namespace doe
