cmake_minimum_required(VERSION 3.15)
project(bpe_core_native LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(core_native
    bpe_tokenizer/core_native/encoder.cpp
    bpe_tokenizer/core_native/bindings.cpp
)

target_compile_options(core_native PRIVATE -O3)

install(TARGETS core_native LIBRARY DESTINATION bpe_tokenizer)
