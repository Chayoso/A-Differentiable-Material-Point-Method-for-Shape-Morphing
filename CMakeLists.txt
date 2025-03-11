cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(polyscope LANGUAGES CXX CUDA)  # CUDA 언어 추가

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}> -O3 -march=native)
endif()

find_package(CUDAToolkit REQUIRED)  # CUDA 툴킷 사용
enable_language(CUDA)

# CUDA 아키텍처 설정
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 86)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

add_subdirectory(polyscope)
add_subdirectory(DiffMPMLib3D)

set(MAIN_FILES 
    DiffMPMAnimator3D/DiffMPMAnimator3D.cpp 
    DiffMPMAnimator3D/MaterialPointImGui.h 
    DiffMPMAnimator3D/MaterialPointImGui.cpp 
    DiffMPMAnimator3D/MPMPointCloudVisualization.h 
    DiffMPMAnimator3D/MPMPointCloudVisualization.cpp 
    DiffMPMAnimator3D/MPMPolyscope.h  
    DiffMPMAnimator3D/MPMPolyscope.cpp 
    DiffMPMAnimator3D/MXImGuiTools.h 
    DiffMPMAnimator3D/MXImGuiTools.cpp
)
set(MESH_FILES 
    PointCloudMesher/PointCloudMesher.cpp
)

add_executable(Mesh ${MESH_FILES})
add_executable(DIFFMPM ${MAIN_FILES})

foreach(TARGET Mesh DIFFMPM)
    target_link_libraries(${TARGET} polyscope OpenMP::OpenMP_CXX DiffMPMLib CUDA::cudart CUDA::cublas)
    target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR}/include)
    target_include_directories(${TARGET} PUBLIC ${CMAKE_SOURCE_DIR}/DiffMPMLib3D)
    set_target_properties(${TARGET} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endforeach()
