set(DESIRED_DOCTEST_VERSION
    2.4.12
    CACHE STRING "Desired doctest version")

enable_testing()

# find_package(doctest ${DESIRED_DOCTEST_VERSION} QUIET)

# if(NOT doctest_FOUND)
FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v${DESIRED_DOCTEST_VERSION})
FetchContent_MakeAvailable(doctest)
# endif()

add_executable(test1 test1.cu)
target_link_libraries(test1 spareribs doctest::doctest)
add_test(NAME test1 COMMAND test1)
