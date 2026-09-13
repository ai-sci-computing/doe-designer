# Catch2 v3: prefer the Homebrew package, fall back to FetchContent.
find_package(Catch2 3 CONFIG QUIET)
if(NOT Catch2_FOUND)
  include(FetchContent)
  FetchContent_Declare(catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.13.0
    SYSTEM)
  FetchContent_MakeAvailable(catch2)
endif()

# One executable per test file keeps failures local and compile times short.
# Each executable is one ctest entry (Catch2's per-case discovery through
# CMake's JSON parser failed intermittently on this machine; the Catch2
# binaries report per case anyway: ./build/tests/<name> [filter]).
function(doe_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE doe::core Catch2::Catch2WithMain)
  target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

doe_add_test(test_smoke)
doe_add_test(test_next_smooth)
doe_add_test(test_fft)
doe_add_test(test_grid)
doe_add_test(test_propagate)
doe_add_test(test_energy)
doe_add_test(test_metrics)
doe_add_test(test_init)
doe_add_test(test_solvers)
doe_add_test(test_quantize)
doe_add_test(test_quantized_solvers)
doe_add_test(test_image)
doe_add_test(test_io)
doe_add_test(test_targets)
doe_add_test(test_pipeline)
doe_add_test(test_cli)
doe_add_test(test_vortex)
doe_add_test(test_render)
doe_add_test(test_viewer_core)

# Web viewer (viewer-web/): the .doev parser and the slice/face/wall mappings are
# plain JavaScript checked with Node's test runner; the parser is cross-checked
# against a file written by doe_design.
find_program(NODE_EXECUTABLE node)
if(NODE_EXECUTABLE)
  add_test(NAME viewer_web
           COMMAND ${NODE_EXECUTABLE} --test "${CMAKE_SOURCE_DIR}/viewer-web/test/doev.test.js"
                   "${CMAKE_SOURCE_DIR}/viewer-web/test/slices.test.js"
                   "${CMAKE_SOURCE_DIR}/viewer-web/test/core_propagate.test.js"
                   "${CMAKE_SOURCE_DIR}/viewer-web/test/core_energy.test.js"
                   "${CMAKE_SOURCE_DIR}/viewer-web/test/core_pipeline.test.js"
                   "${CMAKE_SOURCE_DIR}/viewer-web/test/browser_registry.test.js")
  set_tests_properties(viewer_web PROPERTIES ENVIRONMENT "DOE_DESIGN=$<TARGET_FILE:doe_design>")
endif()
