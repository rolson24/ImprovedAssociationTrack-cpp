cmake_minimum_required(VERSION 3.14)
project(ImprAssoctrack LANGUAGES CXX CUDA) # Add CUDA as a project language


IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")
# set(CMAKE_CXX_FLAGS_DEBUG "/od /Zi /Wall /W4")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra")


set(CMAKE_CXX_STANDARD 17) # Set C++17 standard
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Ensure C++17 is required
set(CMAKE_CXX_EXTENSIONS OFF) # Turn off non-standard extensions

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include FetchContent module
include(FetchContent)

set(EIGEN_VERSION 3.3)
find_package(Eigen3 ${EIGEN_VERSION} QUIET)
if(NOT EIGEN3_FOUND)
  set(BUILD_TESTING OFF CACHE INTERNAL "")
  FetchContent_Declare(eigen
    GIT_REPOSITORY  https://gitlab.com/libeigen/eigen.git
    GIT_TAG         ${EIGEN_VERSION}
    GIT_SHALLOW     ON)
  FetchContent_MakeAvailable(eigen)
  unset(BUILD_TESTING CACHE)
endif()

# Find and include BOOST
find_package(Boost REQUIRED COMPONENTS filesystem)
include_directories(${Boost_INCLUDE_DIRS})

add_library(${PROJECT_NAME} SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ImprAssocTracker.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KalmanFilter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/lapjv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Object.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Rect.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/STrack.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ReID.cpp
  )
target_include_directories(${PROJECT_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen ${Boost_LIBRARIES})

# Find and link CUDA
find_package(CUDA QUIET)

if(NOT ${CUDA_FOUND})
    message(WARNING "CUDA not found, ReID won't be built")
else()
    message(STATUS "CUDA version ${CUDA_VERSION_STRING} found")
    target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})

    target_sources(${PROJECT_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/src/TRT_InferenceEngine/TensorRT_InferenceEngine.cpp
        ${PROJECT_SOURCE_DIR}/src/ReID.cpp)
    target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} nvonnxparser nvinfer)
endif()

# Find and link OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Build the example usage if the 'BUILD_IMPRASSOCTRACK_INFERENCE' variable is set to 'ON'
set(BUILD_IMPRASSOCTRACK_INFERENCE OFF CACHE BOOL "The flag whether to build the inference or not")
if(BUILD_IMPRASSOCTRACK_INFERENCE)
  set_target_properties(ImprAssoctrack PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    OUTPUT_NAME "ImprAssoctrack"
  )

  add_executable(${PROJECT_NAME}_inference 
    ${CMAKE_CURRENT_SOURCE_DIR}/test/example_usage_Impr_Assoc.cpp
  )
  target_link_libraries(${PROJECT_NAME}_inference 
    PRIVATE
    ${PROJECT_NAME}
  )
endif()

# Build the tests if the 'BUILD_IMPRASSOCTRACK_TEST' variable is set to 'ON'
set(BUILD_IMPRASSOCTRACK_TEST OFF CACHE BOOL "The flag whether to build the tests or not")
if(BUILD_IMPRASSOCTRACK_TEST)
  enable_testing()

  add_subdirectory(googletest)

  include_directories(googletest)

  # find_package(GTest QUIET)
  # if(NOT GTest_FOUND)
  #   message(FATAL_ERROR "GTest not found.")
  # endif()
  set_target_properties(ImprAssoctrack PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    OUTPUT_NAME "ImprAssoctrack"
  )

  add_executable(${PROJECT_NAME}_test
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_ImprAssocTracker.cpp
  )


  target_link_libraries(${PROJECT_NAME}_test
    PRIVATE
    Boost::filesystem
    ${PROJECT_NAME}
    gtest
    )
  include(GoogleTest)

  gtest_discover_tests(${PROJECT_NAME}_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/YOLOX_ncnn_palace)
endif()
