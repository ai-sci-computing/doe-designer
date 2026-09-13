#include "doe/image.hpp"

#include "colormap_lut.hpp"
#include "doe/phase.hpp"

#include <png.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numbers>
#include <stdexcept>

namespace doe {
namespace {

struct FileCloser {
    void operator()(std::FILE* f) const { if (f) std::fclose(f); }
};
using FilePtr = std::unique_ptr<std::FILE, FileCloser>;

RGB lut_sample(const std::uint8_t (*lut)[3], double t) {
    t = std::clamp(t, 0.0, 1.0);
    const int i = static_cast<int>(std::lround(t * 255.0));
    return {lut[i][0], lut[i][1], lut[i][2]};
}

}  // namespace

ImageU8 read_png(const std::filesystem::path& path) {
    FilePtr fp(std::fopen(path.string().c_str(), "rb"));
    if (!fp) throw std::runtime_error("read_png: cannot open " + path.string());
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) throw std::runtime_error("read_png: png_create_read_struct failed");
    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, nullptr, nullptr);
        throw std::runtime_error("read_png: png_create_info_struct failed");
    }
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        throw std::runtime_error("read_png: libpng error reading " + path.string());
    }
    png_init_io(png, fp.get());
    png_read_info(png, info);
    // Normalize every input to 8-bit gray / RGB / RGBA.
    const png_byte color = png_get_color_type(png, info);
    const png_byte depth = png_get_bit_depth(png, info);
    if (color == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color == PNG_COLOR_TYPE_GRAY && depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (depth == 16) png_set_strip_16(png);
    png_read_update_info(png, info);

    ImageU8 img;
    img.width = png_get_image_width(png, info);
    img.height = png_get_image_height(png, info);
    img.channels = png_get_channels(png, info);
    if (img.channels == 2) {  // gray + alpha: drop alpha below
        img.channels = 2;
    }
    const std::size_t stride = png_get_rowbytes(png, info);
    std::vector<std::uint8_t> raw(stride * img.height);
    std::vector<png_bytep> rows(img.height);
    for (std::size_t y = 0; y < img.height; ++y) rows[y] = raw.data() + y * stride;
    png_read_image(png, rows.data());
    png_destroy_read_struct(&png, &info, nullptr);

    if (img.channels == 2) {  // gray+alpha -> gray
        img.data.resize(img.width * img.height);
        for (std::size_t k = 0; k < img.data.size(); ++k) img.data[k] = raw[2 * k];
        img.channels = 1;
    } else {
        img.data.assign(raw.begin(), raw.begin() + static_cast<std::ptrdiff_t>(img.width * img.height * img.channels));
    }
    return img;
}

void write_png(const std::filesystem::path& path, const ImageU8& image) {
    if (image.channels != 1 && image.channels != 3 && image.channels != 4)
        throw std::invalid_argument("write_png: channels must be 1, 3 or 4");
    if (image.data.size() != image.width * image.height * image.channels)
        throw std::invalid_argument("write_png: data size does not match width * height * channels");
    FilePtr fp(std::fopen(path.string().c_str(), "wb"));
    if (!fp) throw std::runtime_error("write_png: cannot open " + path.string());
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) throw std::runtime_error("write_png: png_create_write_struct failed");
    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, nullptr);
        throw std::runtime_error("write_png: png_create_info_struct failed");
    }
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        throw std::runtime_error("write_png: libpng error writing " + path.string());
    }
    png_init_io(png, fp.get());
    const int color = image.channels == 1 ? PNG_COLOR_TYPE_GRAY : image.channels == 3 ? PNG_COLOR_TYPE_RGB : PNG_COLOR_TYPE_RGBA;
    png_set_IHDR(png, info, static_cast<png_uint_32>(image.width), static_cast<png_uint_32>(image.height), 8, color,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);
    std::vector<png_bytep> rows(image.height);
    const std::size_t stride = image.width * image.channels;
    for (std::size_t y = 0; y < image.height; ++y) rows[y] = const_cast<png_bytep>(image.data.data() + y * stride);
    png_write_image(png, rows.data());
    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
}

Array2<double> to_gray(const ImageU8& image) {
    Array2<double> a(image.width, image.height);
    for (std::size_t y = 0; y < image.height; ++y)
        for (std::size_t x = 0; x < image.width; ++x) {
            const std::uint8_t* p = &image.data[(y * image.width + x) * image.channels];
            double v;
            if (image.channels == 1) v = p[0];
            else v = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];  // Rec. 601 luma
            a(x, y) = v / 255.0;
        }
    return a;
}

Array2<double> resample(const Array2<double>& src, std::size_t rows, std::size_t cols) {
    if (src.rows == 0 || src.cols == 0 || rows == 0 || cols == 0) throw std::invalid_argument("resample: empty array");
    Array2<double> out(rows, cols);
    const double sx = rows > 1 ? double(src.rows - 1) / double(rows - 1) : 0.0;
    const double sy = cols > 1 ? double(src.cols - 1) / double(cols - 1) : 0.0;
    for (std::size_t i = 0; i < rows; ++i) {
        const double u = i * sx;
        const std::size_t i0 = std::min(static_cast<std::size_t>(u), src.rows - 1), i1 = std::min(i0 + 1, src.rows - 1);
        const double fu = u - double(i0);
        for (std::size_t j = 0; j < cols; ++j) {
            const double v = j * sy;
            const std::size_t j0 = std::min(static_cast<std::size_t>(v), src.cols - 1), j1 = std::min(j0 + 1, src.cols - 1);
            const double fv = v - double(j0);
            out(i, j) = (1 - fu) * ((1 - fv) * src(i0, j0) + fv * src(i0, j1)) + fu * ((1 - fv) * src(i1, j0) + fv * src(i1, j1));
        }
    }
    return out;
}

Array2<double> embed(const Array2<double>& src, std::size_t n) {
    if (src.rows > n || src.cols > n) throw std::invalid_argument("embed: source larger than the window");
    Array2<double> out(n, n, 0.0);
    const std::size_t i0 = (n - src.rows) / 2, j0 = (n - src.cols) / 2;
    for (std::size_t i = 0; i < src.rows; ++i)
        for (std::size_t j = 0; j < src.cols; ++j) out(i0 + i, j0 + j) = src(i, j);
    return out;
}

Array2<double> crop_center(const Array2<double>& a, std::size_t rows, std::size_t cols) {
    if (rows > a.rows || cols > a.cols) throw std::invalid_argument("crop_center: region larger than the array");
    Array2<double> out(rows, cols);
    const std::size_t i0 = (a.rows - rows) / 2, j0 = (a.cols - cols) / 2;
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j) out(i, j) = a(i0 + i, j0 + j);
    return out;
}

Field<double> crop_center_field(const Field<double>& a, std::size_t n) {
    if (n > a.rows || n > a.cols) throw std::invalid_argument("crop_center_field: region larger than the field");
    Field<double> out(n, n);
    const std::size_t i0 = (a.rows - n) / 2, j0 = (a.cols - n) / 2;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) out(i, j) = a(i0 + i, j0 + j);
    return out;
}

RGB colormap_twilight(double t) { return lut_sample(lut::twilight, t); }
RGB colormap_viridis(double t) { return lut_sample(lut::viridis, t); }

RGB hsv_phasor(std::complex<double> z, double amp_max) {
    const double value = amp_max > 0 ? std::clamp(std::abs(z) / amp_max, 0.0, 1.0) : 0.0;
    // hue in [0, 6): phase 0 -> red, 2 pi / 3 -> green, -2 pi / 3 -> blue
    double h = wrap_to_pi(std::arg(z));
    if (h < 0) h += 2 * std::numbers::pi;
    h = h / (2 * std::numbers::pi) * 6.0;
    const int sector = static_cast<int>(std::floor(h)) % 6;
    const double f = h - std::floor(h);
    // saturation 1: p = 0, q = 1 - f, t = f
    double r, g, b;
    switch (sector) {
        case 0: r = 1; g = f; b = 0; break;
        case 1: r = 1 - f; g = 1; b = 0; break;
        case 2: r = 0; g = 1; b = f; break;
        case 3: r = 0; g = 1 - f; b = 1; break;
        case 4: r = f; g = 0; b = 1; break;
        default: r = 1; g = 0; b = 1 - f; break;
    }
    auto q = [value](double c) { return static_cast<std::uint8_t>(std::lround(std::clamp(c * value, 0.0, 1.0) * 255.0)); };
    return {q(r), q(g), q(b)};
}

namespace {
ImageU8 rgb_image(std::size_t rows, std::size_t cols) {
    ImageU8 img;
    img.width = rows;
    img.height = cols;
    img.channels = 3;
    img.data.resize(rows * cols * 3);
    return img;
}
void put(ImageU8& img, std::size_t x, std::size_t y, RGB c) {
    std::uint8_t* p = &img.data[(y * img.width + x) * 3];
    p[0] = c.r;
    p[1] = c.g;
    p[2] = c.b;
}
}  // namespace

ImageU8 render_phase(const Array2<double>& phi) {
    ImageU8 img = rgb_image(phi.rows, phi.cols);
    for (std::size_t i = 0; i < phi.rows; ++i)
        for (std::size_t j = 0; j < phi.cols; ++j)
            put(img, i, j, colormap_twilight((wrap_to_pi(phi(i, j)) + std::numbers::pi) / (2 * std::numbers::pi)));
    return img;
}

ImageU8 render_intensity(const Array2<double>& intensity, bool log_scale, double floor_db) {
    double imax = 0.0;
    for (double v : intensity.data) imax = std::max(imax, v);
    ImageU8 img = rgb_image(intensity.rows, intensity.cols);
    for (std::size_t i = 0; i < intensity.rows; ++i)
        for (std::size_t j = 0; j < intensity.cols; ++j) {
            const double rel = imax > 0 ? std::max(intensity(i, j), 0.0) / imax : 0.0;
            double t;
            if (log_scale) {
                const double db = rel > 0 ? 10.0 * std::log10(rel) : -1e300;
                t = 1.0 + db / floor_db;  // -floor_db .. 0 dB -> 0 .. 1
            } else {
                t = rel;
            }
            put(img, i, j, colormap_viridis(t));
        }
    return img;
}

ImageU8 render_hsv(const Field<double>& v) {
    double amax = 0.0;
    for (const auto& z : v.data) amax = std::max(amax, std::abs(z));
    ImageU8 img = rgb_image(v.rows, v.cols);
    for (std::size_t i = 0; i < v.rows; ++i)
        for (std::size_t j = 0; j < v.cols; ++j) put(img, i, j, hsv_phasor(v(i, j), amax));
    return img;
}

}  // namespace doe
