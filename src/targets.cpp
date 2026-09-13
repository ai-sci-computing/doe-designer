#include "doe/targets.hpp"

#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

namespace doe::targets {
namespace {

// 5 x 7 bitmap font: seven rows of five characters, '#' = ink.
const std::map<char, std::vector<const char*>>& font() {
    static const std::map<char, std::vector<const char*>> f = {
        {' ', {".....", ".....", ".....", ".....", ".....", ".....", "....."}},
        {'A', {".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"}},
        {'B', {"####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."}},
        {'C', {".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."}},
        {'D', {"####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."}},
        {'E', {"#####", "#....", "#....", "####.", "#....", "#....", "#####"}},
        {'F', {"#####", "#....", "#....", "####.", "#....", "#....", "#...."}},
        {'G', {".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".####"}},
        {'H', {"#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"}},
        {'I', {".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."}},
        {'J', {"..###", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."}},
        {'K', {"#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"}},
        {'L', {"#....", "#....", "#....", "#....", "#....", "#....", "#####"}},
        {'M', {"#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"}},
        {'N', {"#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"}},
        {'O', {".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."}},
        {'P', {"####.", "#...#", "#...#", "####.", "#....", "#....", "#...."}},
        {'Q', {".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"}},
        {'R', {"####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"}},
        {'S', {".####", "#....", "#....", ".###.", "....#", "....#", "####."}},
        {'T', {"#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."}},
        {'U', {"#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."}},
        {'V', {"#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."}},
        {'W', {"#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"}},
        {'X', {"#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"}},
        {'Y', {"#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."}},
        {'Z', {"#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"}},
        {'0', {".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."}},
        {'1', {"..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."}},
        {'2', {".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"}},
        {'3', {"#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."}},
        {'4', {"...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."}},
        {'5', {"#####", "#....", "####.", "....#", "....#", "#...#", ".###."}},
        {'6', {"..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."}},
        {'7', {"#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."}},
        {'8', {".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."}},
        {'9', {".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."}},
        {'+', {".....", "..#..", "..#..", "#####", "..#..", "..#..", "....."}},
        {'-', {".....", ".....", ".....", "#####", ".....", ".....", "....."}},
        {'.', {".....", ".....", ".....", ".....", ".....", ".##..", ".##.."}},
        {':', {".....", ".##..", ".##..", ".....", ".##..", ".##..", "....."}},
    };
    return f;
}

}  // namespace

Array2<double> cross_plus_ring(std::size_t n) {
    Array2<double> t(n, n, 0.0);
    const double c = (double(n) - 1.0) / 2.0, s = double(n) / 256.0;
    const double hw = 6 * s, hl = 60 * s, r0 = 80 * s, r1 = 92 * s;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double x = double(i) - c, y = double(j) - c, r = std::hypot(x, y);
            const bool cross = (std::abs(x) < hw && std::abs(y) < hl) || (std::abs(y) < hw && std::abs(x) < hl);
            const bool ring = r > r0 && r < r1;
            t(i, j) = (cross || ring) ? 1.0 : 0.0;
        }
    return t;
}

Array2<double> spot_array(std::size_t n, int nx, int ny, double spot_radius_px, double spacing) {
    if (nx < 1 || ny < 1) throw std::invalid_argument("spot_array: nx, ny >= 1");
    Array2<double> t(n, n, 0.0);
    const double c = (double(n) - 1.0) / 2.0, pitch = spacing * double(n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double x = double(i) - c, y = double(j) - c;
            for (int a = 0; a < nx && t(i, j) == 0.0; ++a)
                for (int b = 0; b < ny; ++b) {
                    const double cx = (a - (nx - 1) / 2.0) * pitch, cy = (b - (ny - 1) / 2.0) * pitch;
                    if (std::hypot(x - cx, y - cy) <= spot_radius_px) { t(i, j) = 1.0; break; }
                }
        }
    return t;
}

Array2<double> disk(std::size_t n, double radius_fraction) {
    Array2<double> t(n, n, 0.0);
    const double c = (double(n) - 1.0) / 2.0, r = radius_fraction * double(n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (std::hypot(double(i) - c, double(j) - c) <= r) t(i, j) = 1.0;
    return t;
}

Array2<double> grating(std::size_t n, int period_px) {
    if (period_px < 2) throw std::invalid_argument("grating: period >= 2");
    Array2<double> t(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        if (static_cast<int>(i % static_cast<std::size_t>(period_px)) < period_px / 2)
            for (std::size_t j = 0; j < n; ++j) t(i, j) = 1.0;
    return t;
}

Array2<double> binary_logo(std::size_t n, const std::string& text) {
    if (text.empty()) throw std::invalid_argument("binary_logo: empty text");
    const auto& f = font();
    for (char ch : text)
        if (!f.count(static_cast<char>(std::toupper(static_cast<unsigned char>(ch)))))
            throw std::invalid_argument(std::string("binary_logo: glyph not in font: ") + ch);
    const std::size_t cols = text.size() * 6 - 1, rows = 7;
    const int scale = std::max(1, static_cast<int>(std::floor(0.8 * double(n) / double(cols))));
    const std::size_t w = cols * static_cast<std::size_t>(scale), h = rows * static_cast<std::size_t>(scale);
    const std::size_t x0 = (n - std::min(w, n)) / 2, y0 = (n - std::min(h, n)) / 2;
    Array2<double> t(n, n, 0.0);
    for (std::size_t k = 0; k < text.size(); ++k) {
        const auto& glyph = f.at(static_cast<char>(std::toupper(static_cast<unsigned char>(text[k]))));
        for (std::size_t r = 0; r < 7; ++r)
            for (std::size_t c = 0; c < 5; ++c) {
                if (glyph[r][c] != '#') continue;
                for (int sx = 0; sx < scale; ++sx)
                    for (int sy = 0; sy < scale; ++sy) {
                        const std::size_t x = x0 + (k * 6 + c) * static_cast<std::size_t>(scale) + static_cast<std::size_t>(sx);
                        const std::size_t y = y0 + r * static_cast<std::size_t>(scale) + static_cast<std::size_t>(sy);
                        if (x < n && y < n) t(x, y) = 1.0;
                    }
            }
    }
    return t;
}

Array2<double> soft_edges(const Array2<double>& target, double sigma_px) {
    if (sigma_px <= 0.0) return target;
    const int half = std::max(1, static_cast<int>(std::ceil(4.0 * sigma_px)));
    std::vector<double> k(static_cast<std::size_t>(2 * half + 1));
    double norm = 0.0;
    for (int d = -half; d <= half; ++d) {
        k[static_cast<std::size_t>(d + half)] = std::exp(-0.5 * (d / sigma_px) * (d / sigma_px));
        norm += k[static_cast<std::size_t>(d + half)];
    }
    for (double& v : k) v /= norm;
    // separable convolution with zero padding outside the image
    Array2<double> tmp(target.rows, target.cols, 0.0), out(target.rows, target.cols, 0.0);
    for (std::size_t i = 0; i < target.rows; ++i)
        for (std::size_t j = 0; j < target.cols; ++j) {
            double s = 0.0;
            for (int d = -half; d <= half; ++d) {
                const long ii = static_cast<long>(i) + d;
                if (ii >= 0 && ii < static_cast<long>(target.rows)) s += k[static_cast<std::size_t>(d + half)] * target(static_cast<std::size_t>(ii), j);
            }
            tmp(i, j) = s;
        }
    for (std::size_t i = 0; i < target.rows; ++i)
        for (std::size_t j = 0; j < target.cols; ++j) {
            double s = 0.0;
            for (int d = -half; d <= half; ++d) {
                const long jj = static_cast<long>(j) + d;
                if (jj >= 0 && jj < static_cast<long>(target.cols)) s += k[static_cast<std::size_t>(d + half)] * tmp(i, static_cast<std::size_t>(jj));
            }
            out(i, j) = std::min(std::max(s, 0.0), 1.0);  // targets are intensities in [0, 1]
        }
    return out;
}

}  // namespace doe::targets
