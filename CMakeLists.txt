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
foreach(t test_hand_eval test_betting test_hand_result eval_exhaustive)
  add_executable(${t} PokerEngine/tests/${t}.cpp)
  target_link_libraries(${t} poker_core Threads::Threads)
  add_test(NAME ${t} COMMAND ${t})
endforeach()

# The trained agent (encoding, mask, forward pass). No fused multiply-add, so that the native and
# WebAssembly builds give the same Q-values bit for bit (web/build.sh passes the same flag).
add_library(poker_agent STATIC PokerEngine/Agent.cpp)
target_link_libraries(poker_agent PUBLIC poker_core)
target_compile_options(poker_agent PRIVATE -ffp-contract=off)

add_executable(test_agent PokerEngine/tests/test_agent.cpp)
target_link_libraries(test_agent poker_agent)
add_test(NAME test_agent COMMAND test_agent ${CMAKE_SOURCE_DIR}/weights/demo mlp-s-seed3)

# The demo's table (web/Table.hpp), the class behind the WebAssembly binding, and the native side
# of web/test.mjs's replay.
add_executable(test_table web/test_table.cpp)
target_include_directories(test_table PRIVATE web)
target_link_libraries(test_table poker_agent)
add_test(NAME test_table COMMAND test_table ${CMAKE_SOURCE_DIR}/weights/demo)

add_executable(bench_engine bench/bench_engine.cpp)
target_link_libraries(bench_engine poker_core)
