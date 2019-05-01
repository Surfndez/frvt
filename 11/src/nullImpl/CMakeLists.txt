cmake_minimum_required (VERSION 3.0)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include)

# Configure built shared libraries in top-level lib directory
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../lib)

find_package (Torch REQUIRED)

# Build the shared libraries
add_library (frvt_11_null_001 SHARED nullimplfrvt11.cpp)

target_link_libraries (frvt_11_null_001 "${TORCH_LIBRARIES}")

