cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(motion_estimation VERSION 0.1)

set(EXE_NAME motion_estimation)

# Set C++ standard to C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)



# opencv
# Find OpenCV
set(OpenCV_DIR "./opencv-4.2.0/build/")
find_package(OpenCV  REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the motion estimation executable
add_executable(${EXE_NAME} src/main.cpp
                           src/motion_estimation.cpp
                           src/helper.cpp)


# Set the include directories
target_include_directories(${EXE_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include
                                              )

target_link_libraries(${EXE_NAME} PUBLIC opencv_core opencv_imgcodecs opencv_highgui)

# Add any additional compilation flags if needed
# target_compile_options(${EXE_NAME} PRIVATE -Wall)

# Set the install targets if needed
# install(TARGETS ${EXE_NAME} DESTINATION bin)
# install(FILES "${PROJECT_SOURCE_DIR}/README.md" DESTINATION share/doc/${EXE_NAME})

