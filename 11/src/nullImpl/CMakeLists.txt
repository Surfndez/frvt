cmake_minimum_required (VERSION 3.0)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Configure built shared libraries in top-level lib directory
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../lib)

find_package (Torch REQUIRED)
find_package (OpenCV REQUIRED)
find_package(InferenceEngine REQUIRED)

include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include
    ${InferenceEngine_INCLUDE_DIRS}
)

file(GLOB algo_src "../algo/*.cpp")

# TensorFlow lib
add_library( tensorflow SHARED IMPORTED PUBLIC )
set_target_properties( tensorflow PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/libtensorflow.so )

# Build the shared libraries
add_library (frvt_11_null_001 SHARED ${algo_src} nullimplfrvt11.cpp)

target_link_libraries (frvt_11_null_001 "${TORCH_LIBRARIES}")
target_link_libraries (frvt_11_null_001 ${OpenCV_LIBS})
target_link_libraries (frvt_11_null_001 ${InferenceEngine_LIBRARIES})
target_link_libraries (frvt_11_null_001 tensorflow)
