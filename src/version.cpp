#include "doe/version.hpp"

#include <fftw3.h>
#include <png.h>

namespace doe {

std::string version() { return "0.1.0"; }

std::string fftw_version_string() { return fftw_version; }

std::string libpng_version_string() { return PNG_LIBPNG_VER_STRING; }

}  // namespace doe
