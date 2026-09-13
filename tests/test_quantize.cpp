// Phase quantization (T5): Wyrowski's stepwise operator and Choi's surrogate.
//
// Wyrowski 1990 (JOSA A 7, 961):
//   eq. (14) levels  {-pi, -pi + D, ..., pi - D},  D = 2 pi / Z
//   eq. (15) direct quantizer Q_Z: nearest level
//   eq. (22) stepwise quantizer Q_Z^(p): a value is projected onto level z only
//            if (z - 0.5 e(p)) D < phi + pi < (z + 0.5 e(p)) D, otherwise kept
//   eq. (23) 0 < e(1) < e(2) < ... < e(P) = 1; the paper's choice for P = 10:
//            0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1
// Choi et al. 2022 (SIGGRAPH), surrogate gradient with Gumbel-Softmax:
//   eq. (5) forward pass uses the hard quantizer q, backward pass the gradient of q_hat
//   eq. (6) q_hat(phi) = sum_l Q_l G_l(score(phi, Q))
//   eq. (7) G_l(z) = exp((z_l + g_l)/tau) / sum_l exp((z_l + g_l)/tau),  g_l ~ Gumbel(0, 1)
//   eq. (8) score_l = sigma(w d(phi, Q_l)) (1 - sigma(w d(phi, Q_l))),  d = signed angular difference
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "doe/quantize.hpp"

#include <cmath>
#include <numbers>
#include <vector>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
constexpr double pi = std::numbers::pi;

namespace {
double wrap(double x) { return std::arg(std::polar(1.0, x)); }
double dist_to_nearest_level(double x, int q) {
    double best = 10;
    for (int z = 0; z < q; ++z) best = std::min(best, std::abs(wrap(x - (-pi + z * 2 * pi / q))));
    return best;
}
}  // namespace

TEST_CASE("Levels: Wyrowski eq. (14)", "[quantize][T5]") {
    doe::Levels L{4};
    CHECK_THAT(L.delta(), WithinRel(pi / 2, 1e-15));
    CHECK_THAT(L.level(0), WithinRel(-pi, 1e-15));
    CHECK_THAT(L.level(1), WithinRel(-pi / 2, 1e-15));
    CHECK_THAT(L.level(2), WithinAbs(0.0, 1e-15));
    CHECK_THAT(L.level(3), WithinRel(pi / 2, 1e-15));
    CHECK(doe::Levels{2}.delta() == pi);
}

TEST_CASE("project_to_levels: nearest level (eq. 15), wrapped, idempotent", "[quantize][T5]") {
    doe::Array2<double> phi(1, 8);
    phi.data = {0.1, 0.9, -3.0, 3.1, -0.9, 2.0, pi, -pi};
    doe::project_to_levels(phi, 4);
    CHECK_THAT(phi.data[0], WithinAbs(0.0, 1e-15));
    CHECK_THAT(phi.data[1], WithinRel(pi / 2, 1e-15));
    CHECK(std::abs(wrap(phi.data[2] - (-pi))) < 1e-15);  // -3.0 is nearest to -pi
    CHECK(std::abs(wrap(phi.data[3] - pi)) < 1e-15);     // 3.1 is nearest to pi = -pi (mod 2 pi)
    CHECK_THAT(phi.data[4], WithinRel(-pi / 2, 1e-15));
    CHECK_THAT(phi.data[5], WithinRel(pi / 2, 1e-15));
    for (double x : phi.data) {
        CHECK(x <= pi);
        CHECK(x > -pi);
        CHECK(dist_to_nearest_level(x, 4) < 1e-15);
    }
    auto again = phi;
    doe::project_to_levels(again, 4);
    CHECK(again.data == phi.data);
    // q = 1: everything collapses to one level
    doe::Array2<double> one(1, 3);
    one.data = {0.3, -2.0, 3.0};
    doe::project_to_levels(one, 1);
    CHECK(dist_to_nearest_level(one.data[0], 1) < 1e-15);
    CHECK(one.data[0] == one.data[1]);
}

TEST_CASE("project_stepwise: Wyrowski eq. (22) captures only the values within +-0.5 e D of a level", "[quantize][T5]") {
    const int q = 4;
    const double D = 2 * pi / q;
    // 400 phases uniformly over (-pi, pi]
    doe::Array2<double> phi(1, 400);
    for (std::size_t k = 0; k < 400; ++k) phi.data[k] = -pi + (double(k) + 0.5) * 2 * pi / 400;
    auto original = phi;

    SECTION("e = 0 leaves everything untouched") {
        CHECK(doe::project_stepwise(phi, q, 0.0) == 0);
        CHECK(phi.data == original.data);
    }
    SECTION("e = 1 is the direct quantizer") {
        CHECK(doe::project_stepwise(phi, q, 1.0) == 400);
        auto direct = original;
        doe::project_to_levels(direct, q);
        for (std::size_t k = 0; k < 400; ++k) CHECK(std::abs(wrap(phi.data[k] - direct.data[k])) < 1e-15);
    }
    SECTION("e = 0.5: exactly the values within 0.25 D of a level move, onto that level") {
        const std::size_t moved = doe::project_stepwise(phi, q, 0.5);
        std::size_t expected = 0;
        for (std::size_t k = 0; k < 400; ++k) {
            const bool inside = dist_to_nearest_level(original.data[k], q) < 0.25 * D;
            expected += inside;
            if (inside)
                CHECK(dist_to_nearest_level(phi.data[k], q) < 1e-15);
            else
                CHECK(phi.data[k] == original.data[k]);
        }
        CHECK(moved == expected);
        CHECK(moved > 150);
        CHECK(moved < 250);  // half of the circle is within the capture intervals
    }
    SECTION("the captured fraction is monotone in e") {
        std::size_t last = 0;
        for (double e : {0.1, 0.3, 0.5, 0.7, 0.9, 1.0}) {
            auto p = original;
            const std::size_t m = doe::project_stepwise(p, q, e);
            CHECK(m >= last);
            last = m;
        }
        CHECK(last == 400);
    }
}

TEST_CASE("wyrowski_epsilon: the paper's table for P = 10 (eq. 23), monotone, ends at 1", "[quantize][T5]") {
    const std::vector<double> table{0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0};
    for (int p = 1; p <= 10; ++p) CHECK_THAT(doe::wyrowski_epsilon(p, 10), WithinAbs(table[p - 1], 1e-15));
    for (int P : {3, 5, 10, 16}) {
        double last = 0;
        for (int p = 1; p <= P; ++p) {
            const double e = doe::wyrowski_epsilon(p, P);
            CHECK(e > last);
            last = e;
        }
        CHECK_THAT(doe::wyrowski_epsilon(P, P), WithinAbs(1.0, 1e-15));
        CHECK(doe::wyrowski_epsilon(1, P) > 0.0);
    }
}

TEST_CASE("capture_fraction: the linear ramp is p/P (Skeren et al. 2002 eq. 11), the table ramp is Wyrowski's", "[quantize][T5]") {
    for (int P : {1, 5, 46, 240}) {
        for (int p = 1; p <= P; ++p)
            CHECK_THAT(doe::capture_fraction(doe::QuantRamp::linear, p, P), WithinAbs(double(p) / double(P), 1e-15));
        CHECK_THAT(doe::capture_fraction(doe::QuantRamp::linear, P, P), WithinAbs(1.0, 1e-15));
    }
    for (int p = 1; p <= 10; ++p)
        CHECK_THAT(doe::capture_fraction(doe::QuantRamp::table, p, 10), WithinAbs(doe::wyrowski_epsilon(p, 10), 1e-15));
    CHECK_THROWS(doe::capture_fraction(doe::QuantRamp::linear, 0, 5));
    CHECK_THROWS(doe::capture_fraction(doe::QuantRamp::linear, 6, 5));
}

TEST_CASE("GumbelSoftmaxQuantizer: without noise and at low temperature q_hat is the nearest level", "[quantize][T5][choi]") {
    doe::GumbelSoftmaxQuantizer Q{4, /*w=*/4.0, /*tau=*/1e-3};
    std::vector<double> zero_noise(4, 0.0);
    for (double phi : {0.1, 1.4, -1.7, 2.9, -0.3}) {
        doe::Array2<double> one(1, 1, phi);
        doe::project_to_levels(one, 4);
        CHECK(std::abs(wrap(Q.value(phi, zero_noise.data()) - one.data[0])) < 1e-6);
    }
}

TEST_CASE("GumbelSoftmaxQuantizer: q_hat is a convex combination of the levels, score is eq. (8)", "[quantize][T5][choi]") {
    doe::GumbelSoftmaxQuantizer Q{4, 4.0, 0.7};
    std::vector<double> noise{0.3, -0.5, 1.2, 0.1};
    for (double phi : {0.1, 1.4, -1.7, 2.9}) {
        const double v = Q.value(phi, noise.data());
        CHECK(v >= -pi);
        CHECK(v <= pi / 2);  // levels of q = 4 span [-pi, pi/2]
    }
    // score_l = sigma(w d)(1 - sigma(w d)) peaks at 1/4 on the level itself
    CHECK_THAT(Q.score(0.0, 2), WithinAbs(0.25, 1e-15));  // level 2 is 0
    CHECK(Q.score(0.5, 2) < 0.25);
    CHECK_THAT(Q.score(0.5, 2), WithinRel(Q.score(-0.5, 2), 1e-15));  // symmetric in the angular difference
    CHECK_THAT(Q.score(pi - 0.1, 0), WithinRel(Q.score(-pi + 0.1, 0), 1e-12));  // difference is taken on the circle
}

TEST_CASE("GumbelSoftmaxQuantizer: derivative matches finite differences (ratio 1 +- 1e-6)", "[quantize][T5][choi]") {
    doe::GumbelSoftmaxQuantizer Q{4, 4.0, 0.5};
    std::vector<double> noise{0.3, -0.5, 1.2, 0.1};
    const double h = 1e-6;
    for (double phi : {0.1, 1.4, -1.7, 2.9, -0.35, 0.8}) {
        const double fd = (Q.value(phi + h, noise.data()) - Q.value(phi - h, noise.data())) / (2 * h);
        const double an = Q.derivative(phi, noise.data());
        REQUIRE(std::abs(an) > 1e-8);
        CHECK_THAT(fd / an, WithinAbs(1.0, 1e-6));
    }
}

TEST_CASE("GumbelSoftmaxQuantizer: Gumbel(0,1) noise is reproducible by seed and has the right mean", "[quantize][choi]") {
    doe::GumbelSoftmaxQuantizer Q{4, 4.0, 0.5};
    auto a = Q.sample_noise(5000, 11);
    auto b = Q.sample_noise(5000, 11);
    auto c = Q.sample_noise(5000, 12);
    REQUIRE(a.rows == 5000);
    REQUIRE(a.cols == 4);
    CHECK(a.data == b.data);
    CHECK(a.data != c.data);
    double mean = 0;
    for (double x : a.data) mean += x;
    mean /= double(a.data.size());
    CHECK_THAT(mean, WithinAbs(0.5772156649, 0.05));  // Euler-Mascheroni constant
}

TEST_CASE("surrogate_gradient: grad_q times dq_hat/dphi elementwise", "[quantize][T5][choi]") {
    doe::GumbelSoftmaxQuantizer Q{4, 4.0, 0.5};
    doe::Array2<double> phi(1, 3), grad_q(1, 3);
    phi.data = {0.1, 1.4, -1.7};
    grad_q.data = {2.0, -1.0, 0.5};
    auto noise = Q.sample_noise(3, 1);
    auto g = doe::surrogate_gradient(grad_q, phi, Q, noise);
    for (std::size_t k = 0; k < 3; ++k)
        CHECK_THAT(g.data[k], WithinRel(grad_q.data[k] * Q.derivative(phi.data[k], &noise(k, 0)), 1e-15));
}
