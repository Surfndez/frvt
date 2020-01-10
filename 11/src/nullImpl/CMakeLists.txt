cmake_minimum_required (VERSION 3.0)

set (MODULE_NAME frvt_11_IntelResearch_001)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-write-strings")

# Configure built shared libraries in top-level lib directory
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../lib)

include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include
    /home/administrator/nist2/openvino/include
    /home/administrator/opencv-4.1.2/include/opencv4
)

file(GLOB algo_src "../algo/*.cpp")

# TensorFlow lib
add_library( tensorflow SHARED IMPORTED PUBLIC )
set_target_properties( tensorflow PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/libtensorflow.so )

# Build the shared libraries
add_library (${MODULE_NAME} SHARED ${algo_src} nullimplfrvt11.cpp)

target_link_libraries (${MODULE_NAME}
    /home/administrator/opencv-4.1.2/lib64/libopencv_core.so
    /home/administrator/opencv-4.1.2/lib64/libopencv_imgproc.so
)
target_link_libraries (${MODULE_NAME} tensorflow)
# target_link_libraries (${MODULE_NAME}  ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/libinference_engine.so)
