cmake_minimum_required(VERSION 3.29)
project(CPLHW1)

set(CMAKE_CXX_STANDARD 20)

find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

add_executable(CPLHW1 main.cpp)
target_link_libraries(CPLHW1 sfml-graphics sfml-window sfml-system)