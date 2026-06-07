add_executable(main main.cu)
target_link_libraries(main spareribs)

add_executable(giulio giulio.cu)
target_link_libraries(giulio spareribs)
