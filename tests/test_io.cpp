// Data files: NumPy .npy (format version 1.0, so results load with numpy.load),
// a minimal JSON writer for the report, CSV for histories and an SVG line plot
// for convergence curves (text output, no plotting library).
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "doe/io.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>

using Catch::Matchers::WithinAbs;

namespace {
std::filesystem::path tmp(const char* name) { return std::filesystem::temp_directory_path() / name; }
std::string slurp(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
}  // namespace

TEST_CASE("npy: double and float arrays round trip with a valid v1.0 header", "[io][npy]") {
    doe::Array2<double> a(3, 5);
    for (std::size_t k = 0; k < 15; ++k) a.data[k] = 0.5 * double(k) - 1.0;
    doe::write_npy(tmp("doe_test_a.npy"), a);
    auto raw = slurp(tmp("doe_test_a.npy"));
    CHECK(raw.substr(0, 6) == std::string("\x93NUMPY", 6));
    CHECK(raw[6] == 1);  // major version 1
    CHECK(raw[7] == 0);  // minor 0
    // header: little-endian float64, C order, shape (3, 5); total header length a multiple of 64
    const std::size_t header_len = static_cast<unsigned char>(raw[8]) | (static_cast<unsigned char>(raw[9]) << 8);
    CHECK((10 + header_len) % 64 == 0);
    const std::string header = raw.substr(10, header_len);
    CHECK(header.find("'descr': '<f8'") != std::string::npos);
    CHECK(header.find("'fortran_order': False") != std::string::npos);
    CHECK(header.find("'shape': (3, 5)") != std::string::npos);
    CHECK(raw.size() == 10 + header_len + 15 * 8);

    auto b = doe::read_npy(tmp("doe_test_a.npy"));
    CHECK(b.rows == 3);
    CHECK(b.cols == 5);
    CHECK(b.data == a.data);

    doe::Array2<float> f(2, 2);
    f.data = {1.5f, -2.f, 0.f, 3.25f};
    doe::write_npy(tmp("doe_test_f.npy"), f);
    CHECK(slurp(tmp("doe_test_f.npy")).find("'descr': '<f4'") != std::string::npos);
    auto g = doe::read_npy(tmp("doe_test_f.npy"));  // reads as double
    CHECK(g.data == std::vector<double>{1.5, -2.0, 0.0, 3.25});
    CHECK_THROWS(doe::read_npy(tmp("doe_missing.npy")));
}

TEST_CASE("Json: objects, arrays, numbers, strings, bools, nesting, escaping", "[io][json]") {
    doe::Json j;
    j.set("name", "cross+ring \"test\"");
    j.set("iters", 300);
    j.set("mu", 0.3);
    j.set("ok", true);
    j.set("nan", std::numeric_limits<double>::quiet_NaN());
    doe::Json m;
    m.set("efficiency", 0.8612345678901234);
    m.set("ncc", 0.957);
    j.set("metrics", m);
    j.set("history", std::vector<double>{1.0, 0.5, 0.25});
    j.set("tags", std::vector<std::string>{"a", "b"});
    const std::string s = j.dump();
    CHECK(s.find("\"name\": \"cross+ring \\\"test\\\"\"") != std::string::npos);
    CHECK(s.find("\"iters\": 300") != std::string::npos);
    CHECK(s.find("\"mu\": 0.3") != std::string::npos);
    CHECK(s.find("\"ok\": true") != std::string::npos);
    CHECK(s.find("\"nan\": null") != std::string::npos);  // JSON has no NaN
    CHECK(s.find("\"efficiency\": 0.8612345678901234") != std::string::npos);  // full precision
    CHECK(s.find("\"history\": [1, 0.5, 0.25]") != std::string::npos);
    CHECK(s.find("\"tags\": [\"a\", \"b\"]") != std::string::npos);
    CHECK(s.find("\"metrics\": {") != std::string::npos);
    // keys keep insertion order
    CHECK(s.find("\"name\"") < s.find("\"iters\""));
    CHECK(s.find("\"iters\"") < s.find("\"metrics\""));
    doe::write_text(tmp("doe_test.json"), s);
    CHECK(slurp(tmp("doe_test.json")) == s);
}

TEST_CASE("csv: named columns", "[io][csv]") {
    doe::write_csv(tmp("doe_test.csv"), {"iter", "energy"}, {{0, 1.0}, {1, 0.5}, {2, 0.25}});
    CHECK(slurp(tmp("doe_test.csv")) == "iter,energy\n0,1\n1,0.5\n2,0.25\n");
}

TEST_CASE("svg_line_plot: valid SVG with axes, labels and one polyline per series", "[io][svg]") {
    doe::SvgPlot plot;
    plot.title = "Convergence";
    plot.x_label = "iteration";
    plot.y_label = "energy";
    plot.log_y = true;
    plot.add_series("Adam", {1.0, 0.5, 0.25, 0.125});
    plot.add_series("GS", {1.0, 0.8, 0.7});
    const std::string svg = plot.render();
    CHECK(svg.rfind("<svg", 0) == 0);
    CHECK(svg.find("</svg>") != std::string::npos);
    CHECK(svg.find("Convergence") != std::string::npos);
    CHECK(svg.find("iteration") != std::string::npos);
    CHECK(svg.find("energy") != std::string::npos);
    CHECK(svg.find(">Adam<") != std::string::npos);
    CHECK(svg.find(">GS<") != std::string::npos);
    std::size_t n_poly = 0;
    for (std::size_t pos = svg.find("<polyline"); pos != std::string::npos; pos = svg.find("<polyline", pos + 1)) ++n_poly;
    CHECK(n_poly == 2);
    doe::write_text(tmp("doe_test.svg"), svg);
    CHECK(slurp(tmp("doe_test.svg")).size() == svg.size());
}

TEST_CASE("svg_line_plot: a dashed series and a color index shared with another series", "[io][svg]") {
    doe::SvgPlot plot;
    plot.add_series("adam shape", {1.0, 0.5, 0.25});
    plot.add_series("adam efficiency", {0.1, 0.05, 0.02}, /*dashed=*/true, /*color=*/0);
    const std::string svg = plot.render();
    CHECK(svg.find("stroke-dasharray") != std::string::npos);
    // both polylines use the same stroke color (the first of the palette); the legend swatches are not polylines
    std::size_t n_first = 0, n_poly = 0;
    for (std::size_t pos = svg.find("<polyline"); pos != std::string::npos; pos = svg.find("<polyline", pos + 1)) {
        ++n_poly;
        const std::string tag = svg.substr(pos, svg.find('>', pos) - pos);
        if (tag.find("stroke=\"#1f77b4\"") != std::string::npos) ++n_first;
    }
    CHECK(n_poly == 2);
    CHECK(n_first == 2);
}

TEST_CASE("SvgPlot: tick labels are round numbers (1, 2, 5 steps on linear axes, powers of ten on a log axis)", "[io][svg]") {
    doe::SvgPlot plot;
    std::vector<double> y(400);
    for (std::size_t k = 0; k < y.size(); ++k) y[k] = 0.8 * std::exp(-double(k) / 60.0) + 0.002;  // 0.802 down to 0.0029
    plot.add_series("e", y);
    const std::string lin = plot.render();
    for (const char* label : {">0<", ">100<", ">200<", ">300<"}) CHECK(lin.find(label) != std::string::npos);
    CHECK(lin.find("79.8") == std::string::npos);
    plot.log_y = true;
    const std::string lg = plot.render();
    for (const char* label : {">0.01<", ">0.1<"}) CHECK(lg.find(label) != std::string::npos);
    CHECK(lg.find("0.799") == std::string::npos);
    CHECK(lg.find("0.183") == std::string::npos);
}

