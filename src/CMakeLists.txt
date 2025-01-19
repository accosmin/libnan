if(APPLE)
    set(base @loader_path)
else()
    set(base $ORIGIN)
endif()

file(RELATIVE_PATH relDir
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
)
set(CMAKE_INSTALL_RPATH ${base} ${base}/${relDir})

# build libraries
make_lib(core)
make_lib(function)
make_lib(solver)
make_lib(machine)
make_lib(linear)

add_subdirectory(core)
add_subdirectory(dataset)
add_subdirectory(datasource)
add_subdirectory(function)
add_subdirectory(gboost)
add_subdirectory(generator)
add_subdirectory(linear)
add_subdirectory(loss)
add_subdirectory(lsearch0)
add_subdirectory(lsearchk)
add_subdirectory(machine)
add_subdirectory(solver)
add_subdirectory(splitter)
add_subdirectory(tuner)
add_subdirectory(wlearner)

# add high-level interface implementations to the associated libraries
target_sources(core PRIVATE
    ${CMAKE_SOURCE_DIR}/include/nano/arch.h
    ${CMAKE_SOURCE_DIR}/include/nano/clonable.h
    ${CMAKE_SOURCE_DIR}/include/nano/configurable.h
    ${CMAKE_SOURCE_DIR}/include/nano/critical.h
    ${CMAKE_SOURCE_DIR}/include/nano/enum.h
    ${CMAKE_SOURCE_DIR}/include/nano/factory.h
    ${CMAKE_SOURCE_DIR}/include/nano/loggable.h
    ${CMAKE_SOURCE_DIR}/include/nano/logger.h
    ${CMAKE_SOURCE_DIR}/include/nano/main.h
    ${CMAKE_SOURCE_DIR}/include/nano/parameter.h
    ${CMAKE_SOURCE_DIR}/include/nano/scalar.h
    ${CMAKE_SOURCE_DIR}/include/nano/string.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/algorithm.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/base.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/dims.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/eigen.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/index.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/integral.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/numeric.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/pprint.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/range.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/stack.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/storage.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/stream.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/tensor.h
    ${CMAKE_SOURCE_DIR}/include/nano/tensor/traits.h
    ${CMAKE_SOURCE_DIR}/include/nano/typed.h
    logger.cpp
    configurable.cpp
    parameter.cpp)

target_sources(function PRIVATE
    ${CMAKE_SOURCE_DIR}/include/nano/function.h
    function.cpp)

target_sources(machine PRIVATE
    ${CMAKE_SOURCE_DIR}/include/nano/dataset.h
    ${CMAKE_SOURCE_DIR}/include/nano/datasource.h
    ${CMAKE_SOURCE_DIR}/include/nano/generator.h
    ${CMAKE_SOURCE_DIR}/include/nano/loss.h
    dataset.cpp
    datasource.cpp
    generator.cpp
    loss.cpp)

target_sources(solver PRIVATE
    ${CMAKE_SOURCE_DIR}/include/nano/lsearch0.h
    ${CMAKE_SOURCE_DIR}/include/nano/lsearchk.h
    ${CMAKE_SOURCE_DIR}/include/nano/solver.h
    lsearch0.cpp
    lsearchk.cpp
    solver.cpp)

target_sources(machine PRIVATE
    ${CMAKE_SOURCE_DIR}/include/nano/feature.h
    ${CMAKE_SOURCE_DIR}/include/nano/learner.h
    ${CMAKE_SOURCE_DIR}/include/nano/splitter.h
    ${CMAKE_SOURCE_DIR}/include/nano/task.h
    ${CMAKE_SOURCE_DIR}/include/nano/tuner.h
    ${CMAKE_SOURCE_DIR}/include/nano/wlearner.h
    feature.cpp
    learner.cpp
    splitter.cpp
    tuner.cpp
    wlearner.cpp)

target_sources(linear PRIVATE
    ${CMAKE_SOURCE_DIR}/include/nano/linear.h
    linear.cpp)

# link libraries
target_link_libraries(core
    PUBLIC Eigen3::Eigen)

if(NOT WIN32)
    target_link_libraries(core
        PUBLIC Threads::Threads
        PRIVATE Threads::Threads)
endif()

target_link_libraries(function PUBLIC core)
target_link_libraries(solver PUBLIC function)
target_link_libraries(machine PUBLIC solver)
target_link_libraries(linear PUBLIC machine)
