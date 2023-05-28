cmake_minimum_required(VERSION 3.10)
project(TPMTracingLibrary VERSION 1.0.0)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O2")

set(ZMQ_LIBRARIES -lzmq)

# Add all source files in the src directory as a shared library
add_library(TPMLibrary SHARED src/tracing.c)

# Add all the include directories
target_include_directories(TPMLibrary PUBLIC 
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/internal
    ${PROJECT_SOURCE_DIR}/include/zmq
)

# Link the ZMQ library
target_link_libraries(TPMLibrary ${ZMQ_LIBRARIES})

# Specify installation directories for the library and headers
install(TARGETS TPMLibrary
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/
        DESTINATION include)
