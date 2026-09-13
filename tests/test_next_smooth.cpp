// T11: next_smooth(n) is 7-smooth and >= n; next_smooth(2048) == 2048.
//
// Why 7-smooth and not the next power of two: FFTW states that it "works most
// efficiently for arrays whose size can be factored into small primes
// (2, 3, 5, and 7)" [fftw_manual]. For an active aperture of 600 the padded
// window is 1200 = 2^4 * 3 * 5^2 instead of 2048.
#include <catch2/catch_test_macros.hpp>

#include "doe/grid.hpp"

namespace {
bool is_7_smooth(int n) {
    if (n < 1) return false;
    for (int p : {2, 3, 5, 7})
        while (n % p == 0) n /= p;
    return n == 1;
}
}  // namespace

TEST_CASE("next_smooth: spec examples (T11)", "[grid][T11]") {
    CHECK(doe::next_smooth(2048) == 2048);  // 2^11
    CHECK(doe::next_smooth(1200) == 1200);  // 2^4 3 5^2
    CHECK(doe::next_smooth(1201) == 1215);  // 3^5 5 is the next 7-smooth number
    CHECK(doe::next_smooth(1) == 1);
    CHECK(doe::next_smooth(0) == 1);
    CHECK(doe::next_smooth(-5) == 1);
}

TEST_CASE("next_smooth: brute-force cross-check up to 5000 (T11)", "[grid][T11]") {
    for (int n = 1; n <= 5000; ++n) {
        int expected = n;
        while (!is_7_smooth(expected)) ++expected;
        REQUIRE(doe::next_smooth(n) == expected);
    }
}
