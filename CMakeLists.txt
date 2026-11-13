cmake_minimum_required(VERSION 3.18...3.31)

# Languages are enabled below only once a C++ compiler is known to exist,
# so a machine without one (e.g. a hosted dashboard) still installs qbt as
# pure Python; qbt.bootstrap's impl="auto" then falls back to NumPy.
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES NONE)

option(QBT_REQUIRE_EXTENSION
       "Fail the build, rather than skip the C++ extension, if it can't be built" OFF)
if(DEFINED ENV{QBT_REQUIRE_EXTENSION} AND "$ENV{QBT_REQUIRE_EXTENSION}")
  set(QBT_REQUIRE_EXTENSION ON)
endif()

include(CheckLanguage)
check_language(CXX)
if(NOT CMAKE_CXX_COMPILER)
  if(QBT_REQUIRE_EXTENSION)
    message(FATAL_ERROR "QBT_REQUIRE_EXTENSION is set but no C++ compiler was found.")
  endif()
  message(WARNING
    "No C++ compiler found: installing qbt without the C++ bootstrap extension. "
    "qbt.bootstrap impl='auto' will use the NumPy implementation.")
  return()
endif()

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# The build-time numpy (pinned in [build-system].requires): its version is
# compiled into the module so qbt.bootstrap can refuse a mismatched runtime
# numpy, and its libnpyrandom supplies the distribution routines.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
    "import numpy, os; print(numpy.__version__); print(numpy.get_include()); print(os.path.join(os.path.dirname(numpy.__file__), 'random', 'lib'))"
  OUTPUT_VARIABLE _numpy_info
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\n" ";" _numpy_info "${_numpy_info}")
list(GET _numpy_info 0 QBT_NUMPY_VERSION)
list(GET _numpy_info 1 QBT_NUMPY_INCLUDE)
list(GET _numpy_info 2 QBT_NUMPY_RANDOM_LIB_DIR)
find_library(QBT_NPYRANDOM NAMES npyrandom PATHS "${QBT_NUMPY_RANDOM_LIB_DIR}"
             NO_DEFAULT_PATH REQUIRED)
message(STATUS "qbt._bootstrap_ext: numpy ${QBT_NUMPY_VERSION}, ${QBT_NPYRANDOM}")

pybind11_add_module(_bootstrap_ext MODULE src/cpp/bootstrap_ext.cpp)
target_include_directories(_bootstrap_ext SYSTEM PRIVATE "${QBT_NUMPY_INCLUDE}")
target_link_libraries(_bootstrap_ext PRIVATE "${QBT_NPYRANDOM}")
if(NOT WIN32)
  target_link_libraries(_bootstrap_ext PRIVATE m)
endif()
target_compile_definitions(_bootstrap_ext PRIVATE
  QBT_NUMPY_BUILD_VERSION="${QBT_NUMPY_VERSION}")
if(MSVC)
  target_compile_options(_bootstrap_ext PRIVATE /W4 /WX)
else()
  target_compile_options(_bootstrap_ext PRIVATE -Wall -Wextra -Werror)
endif()

install(TARGETS _bootstrap_ext LIBRARY DESTINATION qbt)
