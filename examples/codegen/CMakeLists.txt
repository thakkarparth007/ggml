#
# codegen

set(TEST_TARGET codegen)
add_executable(${TEST_TARGET} main.cpp common.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml_utils)

#
# codegen-quantize

set(TEST_TARGET codegen-quantize)
add_executable(${TEST_TARGET} quantize.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml_utils)

# 
# codegen-serve

set(TEST_TARGET codegen-serve)
add_executable(${TEST_TARGET} serve.cpp common.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml_utils)
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
