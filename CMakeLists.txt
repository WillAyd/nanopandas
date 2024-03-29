cmake_minimum_required(VERSION 3.18)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
else()
    add_compile_options(-Wall -Wextra)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

include(FetchContent)
FetchContent_Declare(nanoarrow-project
  GIT_REPOSITORY https://github.com/apache/arrow-nanoarrow.git
  GIT_TAG 4888338903e0dc9867809e3dc7494908f4907d67
)
FetchContent_MakeAvailable(nanoarrow-project)

FetchContent_Declare(utf8proc-project
  GIT_REPOSITORY https://github.com/JuliaStrings/utf8proc
  GIT_TAG v2.9.0
)
# Instead of FetchContent_MakeAvailable(utf8proc-project) we use a custom
# solution that prevents utf8proc from potentially being installed into a
# system location on the host
# See https://mcopik.github.io/blog/2023/cmake/
#FetchContent_MakeAvailable(utf8proc-project)
FetchContent_GetProperties(utf8proc-project)
if(NOT utf8proc-project_POPULATED)
  FetchContent_Populate(utf8proc-project)
  add_subdirectory(${utf8proc-project_SOURCE_DIR} ${utf8proc-project_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()


find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
if (CLANG_TIDY_EXE_FOUND)
  set(CLANG_TIDY_COMMAND "${CLANG_TIDY_EXE}" "-checks=-*,modernize-*")
else()
  message("Could not find clang-tidy installation - checks disabled")
endif()

if (USE_SANITIZERS)
    add_compile_options(-fsanitize=address -fsanitize=undefined)
    add_link_options(-fsanitize=address -fsanitize=undefined)
endif()

add_subdirectory(src/nanopandas)
