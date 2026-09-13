#include "doe/io.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace doe {

// ---------------------------------------------------------------------------
// npy
// ---------------------------------------------------------------------------

template <class T>
void write_npy(const std::filesystem::path& path, const Array2<T>& a) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    const char* descr = std::is_same_v<T, double> ? "<f8" : "<f4";
    std::string header = std::format("{{'descr': '{}', 'fortran_order': False, 'shape': ({}, {}), }}", descr, a.rows, a.cols);
    // total = 10 (magic + version + len) + header must be a multiple of 64; header ends with '\n'
    std::size_t total = 10 + header.size() + 1;
    const std::size_t pad = (64 - total % 64) % 64;
    header.append(pad, ' ');
    header.push_back('\n');
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("write_npy: cannot open " + path.string());
    const std::uint16_t hlen = static_cast<std::uint16_t>(header.size());
    f.write("\x93NUMPY\x01\x00", 8);
    f.put(static_cast<char>(hlen & 0xff));
    f.put(static_cast<char>(hlen >> 8));
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    f.write(reinterpret_cast<const char*>(a.data.data()), static_cast<std::streamsize>(a.data.size() * sizeof(T)));
}

Array2<double> read_npy(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("read_npy: cannot open " + path.string());
    char magic[8];
    f.read(magic, 8);
    if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("read_npy: not an npy file");
    std::size_t hlen = 0;
    if (magic[6] == 1) {
        unsigned char b[2];
        f.read(reinterpret_cast<char*>(b), 2);
        hlen = b[0] | (b[1] << 8);
    } else {
        unsigned char b[4];
        f.read(reinterpret_cast<char*>(b), 4);
        hlen = b[0] | (b[1] << 8) | (b[2] << 16) | (std::size_t(b[3]) << 24);
    }
    std::string header(hlen, '\0');
    f.read(header.data(), static_cast<std::streamsize>(hlen));
    const bool f8 = header.find("'<f8'") != std::string::npos, f4 = header.find("'<f4'") != std::string::npos;
    if (!f8 && !f4) throw std::runtime_error("read_npy: only little-endian float32/float64 supported");
    if (header.find("'fortran_order': True") != std::string::npos) throw std::runtime_error("read_npy: Fortran order not supported");
    const auto sp = header.find("'shape': (");
    if (sp == std::string::npos) throw std::runtime_error("read_npy: no shape");
    std::size_t rows = 0, cols = 0;
    std::istringstream shape(header.substr(sp + 10));
    char comma;
    if (!(shape >> rows >> comma >> cols)) throw std::runtime_error("read_npy: only 2-D arrays supported");
    Array2<double> a(rows, cols);
    if (f8) {
        f.read(reinterpret_cast<char*>(a.data.data()), static_cast<std::streamsize>(rows * cols * 8));
    } else {
        std::vector<float> buf(rows * cols);
        f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(rows * cols * 4));
        for (std::size_t k = 0; k < buf.size(); ++k) a.data[k] = buf[k];
    }
    if (!f) throw std::runtime_error("read_npy: truncated file");
    return a;
}

template void write_npy<float>(const std::filesystem::path&, const Array2<float>&);
template void write_npy<double>(const std::filesystem::path&, const Array2<double>&);

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------

namespace {
std::string json_string(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            default: out += c;
        }
    }
    return out + "\"";
}
std::string json_number(double v) {
    if (!std::isfinite(v)) return "null";
    return std::format("{}", v);  // shortest round-trip representation
}
}  // namespace

void Json::set(const std::string& key, const std::string& value) { items_.emplace_back(key, json_string(value)); }
void Json::set(const std::string& key, double value) { items_.emplace_back(key, json_number(value)); }
void Json::set(const std::string& key, int value) { items_.emplace_back(key, std::to_string(value)); }
void Json::set(const std::string& key, bool value) { items_.emplace_back(key, value ? "true" : "false"); }
void Json::set(const std::string& key, const Json& value) { items_.emplace_back(key, "\x01" + value.dump()); }
void Json::set(const std::string& key, const std::vector<double>& values) {
    std::string s = "[";
    for (std::size_t k = 0; k < values.size(); ++k) s += (k ? ", " : "") + json_number(values[k]);
    items_.emplace_back(key, s + "]");
}
void Json::set(const std::string& key, const std::vector<std::string>& values) {
    std::string s = "[";
    for (std::size_t k = 0; k < values.size(); ++k) s += (k ? ", " : "") + json_string(values[k]);
    items_.emplace_back(key, s + "]");
}

std::string Json::dump(int indent) const {
    const std::string pad(static_cast<std::size_t>(indent + 2), ' '), end(static_cast<std::size_t>(indent), ' ');
    std::string s = "{\n";
    for (std::size_t k = 0; k < items_.size(); ++k) {
        std::string value = items_[k].second;
        if (!value.empty() && value[0] == '\x01') {  // nested object: re-indent
            value.erase(0, 1);
            std::string re;
            for (char c : value) {
                re += c;
                if (c == '\n') re += std::string(static_cast<std::size_t>(indent + 2), ' ');
            }
            value = re;
        }
        s += pad + json_string(items_[k].first) + ": " + value + (k + 1 < items_.size() ? ",\n" : "\n");
    }
    return s + end + "}";
}

// ---------------------------------------------------------------------------
// text, csv, svg
// ---------------------------------------------------------------------------

void write_text(const std::filesystem::path& path, const std::string& text) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("write_text: cannot open " + path.string());
    f << text;
}

void write_csv(const std::filesystem::path& path, const std::vector<std::string>& columns,
               const std::vector<std::vector<double>>& rows) {
    std::string s;
    for (std::size_t k = 0; k < columns.size(); ++k) s += (k ? "," : "") + columns[k];
    s += "\n";
    for (const auto& r : rows) {
        for (std::size_t k = 0; k < r.size(); ++k) s += (k ? "," : "") + json_number(r[k]);
        s += "\n";
    }
    write_text(path, s);
}

void SvgPlot::add_series(const std::string& name, const std::vector<double>& y, bool dashed, int color) {
    std::vector<double> x(y.size());
    for (std::size_t k = 0; k < y.size(); ++k) x[k] = static_cast<double>(k);
    add_series(name, x, y, dashed, color);
}

void SvgPlot::add_series(const std::string& name, const std::vector<double>& x, const std::vector<double>& y,
                         bool dashed, int color) {
    if (x.size() != y.size()) throw std::invalid_argument("SvgPlot: x and y sizes differ");
    if (color < 0) {
        color = 0;
        for (const auto& s : series_) color = std::max(color, s.color + 1);
    }
    series_.push_back({name, x, y, dashed, color});
}

std::string SvgPlot::render() const {
    const double ml = 70, mr = 20, mt = 40, mb = 55;  // margins
    const double pw = width - ml - mr, ph = height - mt - mb;
    double xmin = 0, xmax = 1, ymin = 0, ymax = 1;
    bool first = true;
    for (const auto& s : series_)
        for (std::size_t k = 0; k < s.x.size(); ++k) {
            const double yv = log_y ? (s.y[k] > 0 ? std::log10(s.y[k]) : std::numeric_limits<double>::quiet_NaN()) : s.y[k];
            if (!std::isfinite(yv)) continue;
            if (first) { xmin = xmax = s.x[k]; ymin = ymax = yv; first = false; }
            xmin = std::min(xmin, s.x[k]); xmax = std::max(xmax, s.x[k]);
            ymin = std::min(ymin, yv); ymax = std::max(ymax, yv);
        }
    if (xmax == xmin) xmax = xmin + 1;
    if (ymax == ymin) ymax = ymin + 1;
    auto X = [&](double x) { return ml + (x - xmin) / (xmax - xmin) * pw; };
    auto Y = [&](double y) { return mt + ph - (y - ymin) / (ymax - ymin) * ph; };
    static const char* colors[] = {"#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"};

    std::string s = std::format("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">\n", width, height, width, height);
    s += "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    s += std::format("<text x=\"{}\" y=\"24\" font-family=\"sans-serif\" font-size=\"16\" text-anchor=\"middle\">{}</text>\n", width / 2.0, title);
    // axes
    s += std::format("<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"black\"/>\n", ml, mt + ph, ml + pw, mt + ph);
    s += std::format("<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"black\"/>\n", ml, mt, ml, mt + ph);
    // Round tick positions: a 1-2-5 step giving about five ticks on a linear axis, integer
    // powers of ten on a log axis (2x and 5x added when the range spans fewer than two decades).
    auto linear_ticks = [](double lo, double hi) {
        const double raw = (hi - lo) / 5.0, mag = std::pow(10.0, std::floor(std::log10(raw)));
        const double r = raw / mag, step = (r <= 1.5 ? 1.0 : r <= 3.5 ? 2.0 : r <= 7.5 ? 5.0 : 10.0) * mag;
        std::vector<double> t;
        for (double v = std::ceil(lo / step - 1e-9) * step; v <= hi + 1e-9 * step; v += step) t.push_back(std::abs(v) < 1e-12 * step ? 0.0 : v);
        return t;
    };
    auto log_ticks = [](double lo, double hi) {  // lo, hi in log10 units; returns log10 positions
        std::vector<double> t;
        const bool fine = hi - lo < 2.0;
        for (int e = static_cast<int>(std::floor(lo)); e <= static_cast<int>(std::ceil(hi)); ++e)
            for (double m : (fine ? std::vector<double>{1.0, 2.0, 5.0} : std::vector<double>{1.0})) {
                const double v = e + std::log10(m);
                if (v >= lo - 1e-9 && v <= hi + 1e-9) t.push_back(v);
            }
        return t;
    };
    for (double fx : linear_ticks(xmin, xmax))
        s += std::format("<text x=\"{}\" y=\"{}\" font-family=\"sans-serif\" font-size=\"11\" text-anchor=\"middle\">{:g}</text>\n", X(fx), mt + ph + 16, fx);
    for (double fy : (log_y ? log_ticks(ymin, ymax) : linear_ticks(ymin, ymax))) {
        const double yl = log_y ? std::pow(10.0, fy) : fy;
        s += std::format("<text x=\"{}\" y=\"{}\" font-family=\"sans-serif\" font-size=\"11\" text-anchor=\"end\">{:g}</text>\n", ml - 6, Y(fy) + 4, yl);
        s += std::format("<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#ddd\"/>\n", ml, Y(fy), ml + pw, Y(fy));
    }
    s += std::format("<text x=\"{}\" y=\"{}\" font-family=\"sans-serif\" font-size=\"13\" text-anchor=\"middle\">{}</text>\n", ml + pw / 2, height - 12.0, x_label);
    s += std::format("<text x=\"16\" y=\"{}\" font-family=\"sans-serif\" font-size=\"13\" text-anchor=\"middle\" transform=\"rotate(-90 16 {})\">{}</text>\n", mt + ph / 2, mt + ph / 2, y_label);
    for (std::size_t si = 0; si < series_.size(); ++si) {
        const auto& sr = series_[si];
        std::string pts;
        for (std::size_t k = 0; k < sr.x.size(); ++k) {
            const double yv = log_y ? (sr.y[k] > 0 ? std::log10(sr.y[k]) : std::numeric_limits<double>::quiet_NaN()) : sr.y[k];
            if (!std::isfinite(yv)) continue;
            pts += std::format("{:.1f},{:.1f} ", X(sr.x[k]), Y(yv));
        }
        const char* c = colors[sr.color % 6];
        const char* dash = sr.dashed ? " stroke-dasharray=\"6,4\"" : "";
        s += std::format("<polyline fill=\"none\" stroke=\"{}\" stroke-width=\"1.5\"{} points=\"{}\"/>\n", c, dash, pts);
        s += std::format("<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"3\"{}/>", ml + pw - 150.0,
                         mt + 11.5 + 16.0 * si, ml + pw - 138.0, mt + 11.5 + 16.0 * si, c, dash);
        s += std::format("<text x=\"{}\" y=\"{}\" font-family=\"sans-serif\" font-size=\"12\">{}</text>\n", ml + pw - 134.0, mt + 14.0 + 16.0 * si, sr.name);
    }
    return s + "</svg>\n";
}

}  // namespace doe
