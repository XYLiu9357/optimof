cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(rac-extractor)

find_package (Eigen3 3.4 REQUIRED NO_MODULE)
set(SRC
    # cif-parser.cpp
    main.cpp
    # rac-extractor.cpp
    utils.cpp
)

add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries (${PROJECT_NAME} Eigen3::Eigen)

# Enable memory checks
if(${CMAKE_BUILD_TYPE} STREQUAL Debug)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
    # enable_testing()
    # add_test(NAME Test1 COMMAND rac-extractor test_arg1 test_arg2)
    # add_test(NAME Test2 COMMAND rac-extractor test_arg3 test_arg4)
    # include(CTest)
endif()