# CMakeLists.txt to build CUDA program from hellocuda.cu
cmake_minimum_required(VERSION 2.8)

# CuTV
project(CuTV)

# CUDA
SET(CUDA_SEPARABLE_COMPILATION ON)
SET(CUDA_NVCC_ARCH compute_61)
SET(CUDA_NVCC_CODE sm_61)
SET(CUDA_NVCC_FLAGS -g -gencode arch=${CUDA_NVCC_ARCH},code=${CUDA_NVCC_CODE};--ptxas-options=-v)
find_package(CUDA REQUIRED)

# FreeImage
set(FI_DIR ${CMAKE_SOURCE_DIR}/depends/FreeImage/Dist/x64)
include_directories(${FI_DIR})
if (WIN32)
	find_library(FI_LIBRARY NAME "FreeImage.lib" PATHS "${FI_DIR}")
endif ()
if (FI_LIBRARY)
    set(FI_FOUND TRUE)
	message(STATUS "Found FreeImage at ${FI_LIBRARY}")
endif ()
if (NOT FI_FOUND)
	message(FATAL_ERROR  "FreeImage was not found, change FI_LIBRARY")
endif()

add_subdirectory(util)
add_subdirectory(operators)
add_subdirectory(tv)

cuda_add_executable(CuTV cutv.cu
    ${${PROJECT_NAME}_HDR}
    ${${PROJECT_NAME}_SRC})

# Link Libraries
cuda_add_cublas_to_target(CuTV)
target_link_libraries(CuTV ${FI_LIBRARY}) 

# Copy the DLL
add_custom_command(TARGET CuTV POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${FI_DIR}/FreeImage.dll"
        $<TARGET_FILE_DIR:CuTV>)                 # <--this is out-file path