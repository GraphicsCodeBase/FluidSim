cmake_minimum_required(VERSION 3.21)
project(FluidSmoke LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Language standards
# ---------------------------------------------------------------------------
# nvcc defaults to C++14, which silently differs from the host compiler and
# produces confusing errors later. Pin both explicitly.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 2070 SUPER is Turing -> compute capability 7.5.
# Building for a single architecture keeps compile times down.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
# glfw3 / glad / glm / imgui come from vcpkg (see vcpkg.json).
# CUDA comes from the system toolkit, not vcpkg.
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
add_executable(fluid
    src/main.cpp

    # Layer 1 - Foundation (scaffolding; you should never need to read these)
    src/engine/App.cpp
    src/engine/Shader.cpp
    src/engine/DisplayTexture.cpp

    # Layer 2 - Physics (the actual project; a Phase 0 placeholder for now)
    src/sim/testpattern.cu
)

target_include_directories(fluid PRIVATE src)

target_link_libraries(fluid PRIVATE
    glfw
    glad::glad
    glm::glm
    imgui::imgui
    OpenGL::GL
    CUDA::cudart
)

# Shaders are read from disk at runtime rather than embedded, so you can edit
# a .glsl file and just restart the app - no rebuild.
target_compile_definitions(fluid PRIVATE
    SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/shaders"
)

# Predictable output location regardless of generator or configuration.
set_target_properties(fluid PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY         "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
    CUDA_SEPARABLE_COMPILATION ON
)

# Make Visual Studio launch the right project on F5.
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT fluid)

message(STATUS "CUDA ${CUDAToolkit_VERSION} -> sm_${CMAKE_CUDA_ARCHITECTURES}")
