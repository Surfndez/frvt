cmake_minimum_required(VERSION 2.8)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)

include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/src/include
)

file(GLOB test_src "*.cpp")

# FRVT lib
add_library( frvt SHARED IMPORTED PUBLIC )
set_target_properties( frvt PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/libfrvt_11_IntelRealSense_000.so )

# Build the executable
add_executable (fa_test ${test_src})

target_link_libraries (fa_test ${OpenCV_LIBS})
target_link_libraries (fa_test frvt)
