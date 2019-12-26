cmake_minimum_required(VERSION 2.8)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-write-strings")


include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include
    /home/administrator/opencv-4.1.2/include/opencv4
)

file(GLOB test_src "*.cpp")

# FRVT lib
add_library( frvt SHARED IMPORTED PUBLIC )
set_target_properties( frvt PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/libfrvt_11_IntelResearch_000.so)

# Build the executable
add_executable (fa_test ${test_src})
target_link_libraries (fa_test 
    /home/administrator/opencv-4.1.2/lib64/libopencv_core.so
    /home/administrator/opencv-4.1.2/lib64/libopencv_imgproc.so
    /home/administrator/opencv-4.1.2/lib64/libopencv_imgcodecs.so)
target_link_libraries (fa_test frvt)
