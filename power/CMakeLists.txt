cmake_minimum_required(VERSION 3.10)
project(TPMPowerControl VERSION 1.0.0)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")

# Add all include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/zmq)
include_directories(${PROJECT_SOURCE_DIR}/include/internal)
include_directories(${PROJECT_SOURCE_DIR}/include/monitor)

# Find the zmq and cpufreq libraries
set(ZMQ_LIBRARY -lzmq)
set(CPUFREQ_LIBRARY -lcpufreq)

# Create executable
add_executable(TPMpower src/power.c)

# Link libraries to your executable
target_link_libraries(TPMpower ${ZMQ_LIBRARY} ${CPUFREQ_LIBRARY})
