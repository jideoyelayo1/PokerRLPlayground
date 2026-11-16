cmake_minimum_required(VERSION 3.20)
project(PokerRLPlayground CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

add_library(poker_core STATIC PokerEngine/GameEngine.cpp PokerEngine/Player.cpp)
target_include_directories(poker_core PUBLIC PokerEngine)

enable_testing()
foreach(t test_hand_eval test_betting eval_exhaustive)
  add_executable(${t} PokerEngine/tests/${t}.cpp)
  target_link_libraries(${t} poker_core Threads::Threads)
  add_test(NAME ${t} COMMAND ${t})
endforeach()
