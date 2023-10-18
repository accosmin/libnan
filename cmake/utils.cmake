# function to copy runtime DLLs for the given target (needed for shared library dependencies on MSVC)
# NB: from https://stackoverflow.com/questions/14089284
function(copy_runtime_dlls TARGET)
    get_property(already_applied TARGET "${TARGET}" PROPERTY _copy_runtime_dlls_applied)

    if (MSVC AND BUILD_SHARED_LIBS AND NOT already_applied)
        add_custom_command(
            TARGET "${TARGET}" POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -v -E copy
                "$<TARGET_RUNTIME_DLLS:${TARGET}>" "$<TARGET_FILE_DIR:${TARGET}>"
            COMMAND_EXPAND_LISTS
        )

        set_property(TARGET "${TARGET}" PROPERTY _copy_runtime_dlls_applied 1)
    endif ()
endfunction()

# function to create a library
function(make_lib lib)
    add_library(${lib})
    target_sources(${lib} PRIVATE ${ARGN})
    add_library(NANO::${lib} ALIAS ${lib})
    target_include_directories(${lib}
        PUBLIC
            $<INSTALL_INTERFACE:include>
            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(${lib}
        PUBLIC Eigen3::Eigen)
    if(NOT WIN32)
        target_link_libraries(${lib}
            PUBLIC Threads::Threads
            PRIVATE Threads::Threads)
    endif()
endfunction()

# function to create a unit test application
function(make_test test)
    add_executable(${test} ${test}.cpp)
    target_include_directories(${test}
        SYSTEM PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/test>
    )
    target_link_libraries(${test}
        PRIVATE ${ARGN}
    )

    if (NANO_ENABLE_LLVM_COV)
        # FIXME: Not working on Windows!
        add_test(
            NAME ${test}
            COMMAND bash -c "rm -f ${test}.profraw ${test}.profdata; \
                    LLVM_PROFILE_FILE=${test}.profraw ${CMAKE_CURRENT_BINARY_DIR}/${test}; \
                    llvm-profdata merge -sparse ${test}.profraw -o ${test}.profdata"
        )
    else()
        add_test(${test} ${test})
    endif()
endfunction()

# function to create an executable
function(make_app app)
    add_executable(${app} ${app}.cpp)
    target_include_directories(${app}
        PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
    target_link_libraries(${app}
        PRIVATE ${ARGN})
    install(TARGETS ${app}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction()
