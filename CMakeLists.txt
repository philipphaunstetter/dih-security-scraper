cmake_minimum_required(VERSION 3.20)
project(YourApp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check for CUDA on Windows
if(WIN32)
    find_package(CUDA)
    if(CUDA_FOUND)
        add_definitions(-DUSE_CUDA)
    endif()
endif()

add_executable(${PROJECT_NAME} src/main.cpp)

# Link CUDA if found
if(WIN32 AND CUDA_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)
endif() 