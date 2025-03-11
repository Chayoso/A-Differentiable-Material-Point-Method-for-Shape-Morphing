cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(DiffMPMLib LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)

# 라이브러리 추가
add_library(DiffMPMLib STATIC
    pch.h
    BackPropagation.h
    BackPropagation.cpp
    cereal_eigen.h
    CompGraph.h
    CompGraph.cpp
    Elasticity.h
    Elasticity.cpp
    ForwardSimulation.h
    ForwardSimulation.cpp
    framework.h
    GeometryLoading.h
    GeometryLoading.cpp
    Grid.h
    Grid.cpp
    GridNode.h
    GridNode.cpp
    Interpolation.h
    MaterialPoint.h
    MaterialPoint.cpp
    MultiThreadForwardSimulation.h
    MultiThreadForwardSimulation.cpp
    PointCloud.h
    PointCloud.cpp
    SphereUnionSurfacing.h
    SphereUnionSurfacing.cpp
    Tensor3x3x3x3.h
    Tensor3x3x3x3.cpp
    Backprop.cuh
    Backprop.cu
)

target_include_directories(DiffMPMLib PUBLIC
    ${CMAKE_SOURCE_DIR}/PointCloudMesher
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(DiffMPMLib PUBLIC CUDA::cudart CUDA::cublas)

set_target_properties(DiffMPMLib PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 14
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON  # 심볼 해결 활성화
    CUDA_ARCHITECTURES "52"
)

if(MSVC)
    target_compile_options(DiffMPMLib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/MP>)
    target_compile_options(DiffMPMLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:/MP>)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
