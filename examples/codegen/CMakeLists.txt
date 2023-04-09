#
# codegen

set(TEST_TARGET codegen)
add_executable(${TEST_TARGET} main.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml_utils)

#
# codegen-quantize

set(TEST_TARGET codegen-quantize)
add_executable(${TEST_TARGET} quantize.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ggml_utils)
