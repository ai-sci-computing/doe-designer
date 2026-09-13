#include "doe/vortex.hpp"

#include "doe/phase.hpp"

#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>

namespace doe {

template <class T>
Array2<int> vortex_charge_map(const Field<T>& u) {
    if (u.rows < 2 || u.cols < 2) throw std::invalid_argument("vortex_charge_map: field too small");
    Array2<int> q(u.rows - 1, u.cols - 1, 0);
    auto ph = [&](std::size_t i, std::size_t j) { return std::arg(std::complex<double>(u(i, j))); };
    auto nz = [&](std::size_t i, std::size_t j) { return std::abs(std::complex<double>(u(i, j))) > 0.0; };
    for (std::size_t i = 0; i + 1 < u.rows; ++i)
        for (std::size_t j = 0; j + 1 < u.cols; ++j) {
            if (!(nz(i, j) && nz(i + 1, j) && nz(i + 1, j + 1) && nz(i, j + 1))) continue;
            const double a = ph(i, j), b = ph(i + 1, j), c = ph(i + 1, j + 1), d = ph(i, j + 1);
            // Goldstein-Zebker-Werner residue: wrapped differences around the loop, in cycles.
            const double s = wrap_to_pi(b - a) + wrap_to_pi(c - b) + wrap_to_pi(d - c) + wrap_to_pi(a - d);
            q(i, j) = static_cast<int>(std::lround(s / (2.0 * std::numbers::pi)));
        }
    return q;
}

std::size_t vortex_count(const Array2<int>& charges) {
    std::size_t n = 0;
    for (int c : charges.data) n += (c != 0);
    return n;
}

template <class T>
double vortex_density(const Field<T>& u, const Grid& grid) {
    const auto q = vortex_charge_map(u);
    const double side = static_cast<double>(u.rows - 1) * grid.pitch;
    const double side2 = static_cast<double>(u.cols - 1) * grid.pitch;
    return static_cast<double>(vortex_count(q)) / (side * side2);
}

template Array2<int> vortex_charge_map<float>(const Field<float>&);
template Array2<int> vortex_charge_map<double>(const Field<double>&);
template double vortex_density<float>(const Field<float>&, const Grid&);
template double vortex_density<double>(const Field<double>&, const Grid&);

}  // namespace doe
