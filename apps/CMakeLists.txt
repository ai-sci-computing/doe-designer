add_executable(doe_design doe_design.cpp)
target_link_libraries(doe_design PRIVATE doe::core)
target_compile_options(doe_design PRIVATE -Wall -Wextra -Wpedantic -Wshadow)

# ---------------------------------------------------------------------------
# doe_view: the animated wavefront box (GLFW + OpenGL 3.3 core + Dear ImGui)
# ---------------------------------------------------------------------------
if(DOE_BUILD_VIEWER)
  find_package(glfw3 3.3 QUIET)
  find_package(OpenGL QUIET)
  find_package(glm QUIET)
  if(NOT glfw3_FOUND OR NOT OpenGL_FOUND OR NOT glm_FOUND)
    message(WARNING "doe_view needs glfw3, OpenGL and glm (brew install glfw glm); skipping the viewer")
  else()
    include(FetchContent)
    FetchContent_Declare(imgui
      GIT_REPOSITORY https://github.com/ocornut/imgui.git
      GIT_TAG        v1.91.9
      GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(imgui)
    add_library(imgui STATIC
      ${imgui_SOURCE_DIR}/imgui.cpp
      ${imgui_SOURCE_DIR}/imgui_draw.cpp
      ${imgui_SOURCE_DIR}/imgui_tables.cpp
      ${imgui_SOURCE_DIR}/imgui_widgets.cpp
      ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
      ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
    target_include_directories(imgui SYSTEM PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
    target_link_libraries(imgui PUBLIC glfw OpenGL::GL)
    target_compile_definitions(imgui PUBLIC GL_SILENCE_DEPRECATION)

    add_executable(doe_view doe_view.cpp)
    target_link_libraries(doe_view PRIVATE doe::core imgui glfw OpenGL::GL glm::glm)
    target_compile_options(doe_view PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
    target_compile_definitions(doe_view PRIVATE GL_SILENCE_DEPRECATION)
  endif()
endif()
