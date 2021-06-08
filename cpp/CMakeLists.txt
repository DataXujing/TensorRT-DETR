cmake_minimum_required(VERSION 3.10)

project(detr)

# opencv
set(OpenCV_DIR /workspace/opencv-4.5.2/build)
find_package(OpenCV REQUIRED)

# ${var}表示引用这个变量
find_package(CUDA  REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TensorRT_INCLUDE_DIRS})
# find_library(CUDA)
# find_library(NVINFER NAMES nvinfer)
# find_library(NVPARSERS NAMES nvparsers)
# find_library(NVONNXPARSERS NAMES nvonnxparser)

find_library(NVINFER NAMES nvinfer)
find_library(NVPARSERS NAMES nvparsers)
find_library(NVONNXPARSERS NAMES nvonnxparser)

find_library(CUDNN_LIBRARY
  NAMES libcudnn.so${__cudnn_ver_suffix} libcudnn${__cudnn_ver_suffix}.dylib ${__cudnn_lib_win_name}
  PATHS $ENV{LD_LIBRARY_PATH} ${__libpath_cudart} ${CUDNN_ROOT_DIR} ${PC_CUDNN_LIBRARY_DIRS} ${CMAKE_INSTALL_PREFIX}
  PATH_SUFFIXES lib lib64 bin
  DOC "CUDNN library."
)


file(GLOB_RECURSE _HEAD ${CMAKE_CURRENT_LIST_DIR}/src/*.h 
                        ${CMAKE_CURRENT_LIST_DIR}/src/*.cuh
)

file(GLOB _SRC ${CMAKE_CURRENT_LIST_DIR}/src/*.cc 
               ${CMAKE_CURRENT_LIST_DIR}/src/*.cu
)


add_executable (${PROJECT_NAME} ${_SRC} ${_HEAD})
target_link_libraries(${PROJECT_NAME}
                      ${NVINFER}
                      ${NVONNXPARSERS}
                      ${CUDA_LIBRARIES}
                      ${CUDA_CUBLAS_LIBRARIES}
                      ${CUDNN_LIBRARY}
                      ${OpenCV_LIBS})

message(STATUS "cmake success!!!  DETR by xj")
