cmake_minimum_required(VERSION 3.25)
project(doe VERSION 0.1.0 LANGUAGES CXX
        DESCRIPTION "Phase-only DOE design from a target image and wavefront-box visualization")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(DOE_BUILD_TESTS  "Build the Catch2 test suite"            ON)
option(DOE_BUILD_DOCS   "Add the Doxygen 'docs' target"          OFF)
option(DOE_BUILD_VIEWER "Build the OpenGL/ImGui viewer doe_view"  ON)
option(DOE_FFTW_MEASURE "Plan FFTs with FFTW_MEASURE (faster, non-deterministic plans)" OFF)

# ---------------------------------------------------------------------------
# System dependencies (Homebrew on macOS): FFTW (double + single), libpng
# ---------------------------------------------------------------------------
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3  REQUIRED IMPORTED_TARGET fftw3)
pkg_check_modules(FFTW3F REQUIRED IMPORTED_TARGET fftw3f)
find_library(FFTW3_THREADS fftw3_threads HINTS ${FFTW3_LIBRARY_DIRS} REQUIRED)
find_library(FFTW3F_THREADS fftw3f_threads HINTS ${FFTW3F_LIBRARY_DIRS} REQUIRED)
find_package(PNG REQUIRED)
if(APPLE AND NOT OpenMP_ROOT AND EXISTS /opt/homebrew/opt/libomp)
  set(OpenMP_ROOT /opt/homebrew/opt/libomp)  # Homebrew libomp for Apple clang
endif()
find_package(OpenMP)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(doe_core
  src/version.cpp
  src/grid.cpp
  src/fft.cpp
  src/propagate.cpp
  src/energy.cpp
  src/metrics.cpp
  src/init.cpp
  src/solvers.cpp
  src/quantize.cpp
  src/image.cpp
  src/io.cpp
  src/targets.cpp
  src/pipeline.cpp
  src/cli.cpp
  src/vortex.cpp
  src/render.cpp
  src/viewer_core.cpp
)
target_include_directories(doe_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(doe_core PUBLIC ${FFTW3_THREADS} ${FFTW3F_THREADS} PkgConfig::FFTW3 PkgConfig::FFTW3F PNG::PNG)
target_compile_options(doe_core PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
if(OpenMP_CXX_FOUND)
  target_link_libraries(doe_core PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(doe_core PUBLIC DOE_HAVE_OPENMP=1)
  message(STATUS "doe_core: OpenMP enabled for element-wise loops")
endif()
if(DOE_FFTW_MEASURE)
  target_compile_definitions(doe_core PUBLIC DOE_FFTW_MEASURE=1)
endif()
add_library(doe::core ALIAS doe_core)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(DOE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Applications
# ---------------------------------------------------------------------------
add_subdirectory(apps)

# ---------------------------------------------------------------------------
# Documentation (Doxygen, MathJax, README as main page, bibliography by key only)
# ---------------------------------------------------------------------------
if(DOE_BUILD_DOCS)
  find_package(Doxygen REQUIRED)
  set(DOXYGEN_PROJECT_NAME "doe-designer")
  set(DOXYGEN_PROJECT_BRIEF "${PROJECT_DESCRIPTION}")
  set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/docs")
  set(DOXYGEN_GENERATE_LATEX NO)
  set(DOXYGEN_USE_MATHJAX YES)
  set(DOXYGEN_MATHJAX_VERSION MathJax_3)
  set(DOXYGEN_MATHJAX_RELPATH "https://cdn.jsdelivr.net/npm/mathjax@3")
  set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
  set(DOXYGEN_CITE_BIB_FILES "${CMAKE_CURRENT_SOURCE_DIR}/docs/references.bib")
  set(DOXYGEN_HTML_EXTRA_STYLESHEET "${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen-extra.css")
  set(DOXYGEN_HTML_COLORSTYLE LIGHT)
  set(DOXYGEN_HTML_COLORSTYLE_HUE 200)
  set(DOXYGEN_HAVE_DOT NO)
  set(DOXYGEN_EXTRACT_ALL NO)
  set(DOXYGEN_WARN_IF_UNDOCUMENTED YES)
  set(DOXYGEN_WARN_AS_ERROR FAIL_ON_WARNINGS)
  set(DOXYGEN_EXCLUDE_PATTERNS "*/build/*" "*/_deps/*")
  set(DOXYGEN_IMAGE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/docs")
  # The manual is written for pandoc/LaTeX ($...$ math, YAML title). Doxygen
  # gets a converted copy: front matter stripped, $...$ -> \f$...\f$.
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/docs/manual.md" _manual)
  string(REGEX REPLACE "^---\ntitle: ([^\n]*)\n---\n" "# \\1 {#manual}\n" _manual "${_manual}")
  string(REGEX REPLACE "\\$([^$]+)\\$" "\\\\f$\\1\\\\f$" _manual "${_manual}")
  file(WRITE "${CMAKE_BINARY_DIR}/docs/manual.md" "${_manual}")
  # The README (GitHub Markdown with $...$ math and links to Markdown files)
  # gets the same treatment; links to files outside the Doxygen input become
  # plain names, the manual link becomes a page reference.
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/README.md" _readme)
  string(REGEX REPLACE "\\$([^$]+)\\$" "\\\\f$\\1\\\\f$" _readme "${_readme}")
  string(REPLACE "](docs/manual.md)" "](@ref manual)" _readme "${_readme}")
  file(WRITE "${CMAKE_BINARY_DIR}/docs/README.md" "${_readme}")
  set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${CMAKE_BINARY_DIR}/docs/README.md")
  doxygen_add_docs(docs
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_BINARY_DIR}/docs/README.md"
    "${CMAKE_CURRENT_SOURCE_DIR}/docs/references.md"
    "${CMAKE_BINARY_DIR}/docs/manual.md"
    COMMENT "Generating API documentation with Doxygen")
endif()

# ---------------------------------------------------------------------------
# Manual: docs/manual.md -> docs/manual.tex (pandoc) -> docs/manual.pdf (xelatex)
#   cmake --build build --target manual_pdf
# ---------------------------------------------------------------------------
find_program(PANDOC pandoc)
find_program(XELATEX xelatex)
if(PANDOC AND XELATEX)
  add_custom_target(manual_pdf
    COMMAND ${PANDOC} manual.md -s -o manual.tex --pdf-engine=xelatex -V geometry:margin=2cm -V fontsize=10pt
            -V colorlinks=true --toc
    COMMAND ${XELATEX} -interaction=nonstopmode -halt-on-error manual.tex > /dev/null
    COMMAND ${XELATEX} -interaction=nonstopmode -halt-on-error manual.tex > /dev/null
    COMMAND ${CMAKE_COMMAND} -E rm -f manual.aux manual.log manual.out manual.toc
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/docs"
    COMMENT "Rendering docs/manual.md -> manual.tex -> manual.pdf")
endif()
