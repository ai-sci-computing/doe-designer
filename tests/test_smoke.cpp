// Infrastructure smoke test: the core library links against FFTW (double and
// single precision) and libpng, and Catch2 is wired into ctest.
#include <catch2/catch_test_macros.hpp>

#include "doe/version.hpp"

#include <fftw3.h>

#include <string_view>

TEST_CASE("library reports its version and back ends", "[smoke]") {
    CHECK(doe::version() == "0.1.0");
    CHECK_FALSE(doe::fftw_version_string().empty());
    CHECK_FALSE(doe::libpng_version_string().empty());
}

TEST_CASE("single-precision FFTW is linked", "[smoke]") {
    // fftwf_ symbols come from libfftw3f; a link error here means the
    // float build of FFTW is missing.
    CHECK(std::string_view(fftwf_version).starts_with("fftw-3"));
}
