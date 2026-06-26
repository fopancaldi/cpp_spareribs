add_executable(main main.cu)
target_link_libraries(main spareribs)

add_executable(spike spike.cu)
target_link_libraries(spike spareribs)

add_executable(orbit orbit.cu)
target_link_libraries(orbit spareribs)
# if(FALSE) target_compile_options(orbit PRIVATE "-fsanitize=address")
# target_link_libraries(orbit asan) endif()
